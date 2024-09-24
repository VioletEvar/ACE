import torch
import pytorch_lightning as pl
from typing import Any, Dict, Mapping, Tuple

from yacs.config import CfgNode

from ..utils import SkeletonRenderer, MeshRenderer
from ..utils.geometry import aa_to_rotmat, perspective_projection
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads import build_mano_head
from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss, InterhandJLoss, InterhandVLoss, MaxMSELoss
from . import MANO

from .components import rat
from .components.sir import SIR, initialize_sir_parameters, MLP
from .components.cropped_image import *

log = get_pylogger(__name__)

class HAMER(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup HAMER model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg

        # 在这里定义rat的参数设置
        self.rat = rat(cfg)
        # 初始化sir
        #self.sir = sir(cfg)
        self.sir = SIR(
            input_dim=cfg.MODEL.SIR.INPUT_DIM,
            hidden_dim=cfg.MODEL.SIR.HIDDEN_SIZE,
            num_heads=cfg.MODEL.SIR.NUM_HEADS,
            num_layers=cfg.MODEL.SIR.NUM_LAYERS,
            output_dim=cfg.MODEL.SIR.OUTPUT_DIM,
        )
        self.mlp = MLP()
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            log.info(f'Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
            self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['state_dict'])
            # Load the pre-trained weights with strict=False
            # self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['state_dict'], strict=False)
            # self.sir.apply(initialize_sir_parameters)


        # Create MANO head
        self.mano_head = build_mano_head(cfg)

        # Create discriminator GAN-based transformer
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            self.discriminator = Discriminator()

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.mano_parameter_loss = ParameterLoss()
        # self.interhand_j_loss = InterhandJLoss(loss_type='l2')

        # Instantiate MANO model
        mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.mano.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

    def get_parameters(self):
        # TODO The meaning of all_params is not clear, should be modified after, The Params should be all put into the optimizer
        all_params = list(self.mano_head.parameters())
        all_params += list(self.backbone.parameters())
        # all_params += list(self.rat.parameters())
        return all_params

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]

        optimizer = torch.optim.AdamW(params=param_groups,
                                        # lr=self.cfg.TRAIN.LR,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                            lr=self.cfg.TRAIN.LR,
                                            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

        return optimizer, optimizer_disc

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        images = batch['img']
        conditioning_feats = self.backbone(images[:,:,:,32:-32])
        lh_box = batch['bbox']
        rh_box = batch['other_hand_bbox']
        batch_size = images.shape[0]
        # Determine the device: Use GPU if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rh_rat_feats = []
        lh_rat_feats = []

        for i in range(batch_size):

            # 获取第 i 个样本的 lh_bbox 和 rh_bbox
            lh_box_np = np.array([
                lh_box[0][i].cpu().item(),  # 提取 x_min
                lh_box[1][i].cpu().item(),  # 提取 y_min
                lh_box[2][i].cpu().item(),  # 提取 x_max
                lh_box[3][i].cpu().item()   # 提取 y_max
            ])
            # print({f'bbox:{lh_box_np}'})
            rh_box_np = np.array([
                rh_box[0][i].cpu().item(),  # 提取 x_min
                rh_box[1][i].cpu().item(),  # 提取 y_min
                rh_box[2][i].cpu().item(),  # 提取 x_max
                rh_box[3][i].cpu().item()   # 提取 y_max
            ])
            # print({f'other hand bbox:{rh_box_np}'})
            rh_rat_feats_tmp, lh_rat_feats_tmp = self.rat(lh_box_np,rh_box_np)
            # rh_rat_feats, lh_rat_feats = torch.stack(rh_rat_feats_tmp, rh_rat_feats), torch.stack(lh_rat_feats, lh_rat_feats_tmp)
            # Append the features to the lists
            rh_rat_feats.append(rh_rat_feats_tmp.to(device))
            lh_rat_feats.append(lh_rat_feats_tmp.to(device))

        expected_feature_shape = (batch_size, 192, 1280)
        if rh_rat_feats:
            rh_rat_feats = torch.stack(rh_rat_feats)
        else:
            # Handle the case where no right-hand features are found
            rh_rat_feats = torch.empty((0, *expected_feature_shape), dtype=torch.float32)

        if lh_rat_feats:
            lh_rat_feats = torch.stack(lh_rat_feats)
        else:
            # Handle the case where no left-hand features are found
            lh_rat_feats = torch.empty((0, *expected_feature_shape), dtype=torch.float32)

        # Concatenate the tensors along the last dimension
        rh_rat_feats = rh_rat_feats.to(device)
        rh_rat_feats = torch.nan_to_num(rh_rat_feats, nan=0.0)
        lh_rat_feats = lh_rat_feats.to(device)
        lh_rat_feats = torch.nan_to_num(lh_rat_feats, nan=0.0)
        
        B, C, H, W = conditioning_feats.shape
        conditioning_feats = conditioning_feats.view(B, H * W, C)
        B, H, W, C = rh_rat_feats.shape
        rh_rat_feats = rh_rat_feats.view(B, H * W, C)
        lh_rat_feats = lh_rat_feats.view(B, H * W, C)
        ult_rat_feats = torch.concat((rh_rat_feats, lh_rat_feats), dim=2)
        ult_feats = torch.concat((conditioning_feats, ult_rat_feats), dim=2)
        ult_feats = torch.nan_to_num(ult_feats, nan=0.0)
        # print(f'shape of correct ult feats:{ult_feats.shape}')
        # print(f'correct ult feats:{ult_feats}')
        """
        The Ultimate features keeps the dimension of 768*2560
        """


        sir_token = self.sir(ult_feats)

        pred_mano_params, pred_cam, _ = self.mano_head(sir_token)
        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam
        # print(f'pred cam:{pred_cam}')
        output['pred_mano_params'] = {k: v.clone() for k,v in pred_mano_params.items()}
        # print(f'pred mano params:{pred_mano_params}')

        # Compute camera translation
        device = pred_mano_params['hand_pose'].device
        dtype = pred_mano_params['hand_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_mano_params['global_orient'] = pred_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['hand_pose'] = pred_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['betas'] = pred_mano_params['betas'].reshape(batch_size, -1)
        mano_output = self.mano(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        # print(f'pred 3d keyp:{pred_keypoints_3d}')
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        # print(f'pred 2d keyp:{pred_keypoints_2d}')
        # TODO compute InterHand Params pred_inter_hand()
        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        pred_mano_params = output['pred_mano_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']



        batch_size = pred_mano_params['hand_pose'].shape[0]
        device = pred_mano_params['hand_pose'].device
        dtype = pred_mano_params['hand_pose'].dtype

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_mano_params = batch['mano_params']
        has_mano_params = batch['has_mano_params']
        is_axis_angle = batch['mano_params_is_axis_angle']



        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0)
        # loss_interhand_j_loss = self.interhand_j_loss(pred_keypoints_left, pred_keypoints_right, gt_left_3d, gt_right_3d)

        # Compute loss on MANO parameters
        loss_mano_params = {}
        for k, pred in pred_mano_params.items():
            gt = gt_mano_params[k].view(batch_size, -1)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_mano_params[k]
            loss_mano_params[k] = self.mano_parameter_loss(pred.reshape(batch_size, -1), gt.reshape(batch_size, -1), has_gt)

        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d+\
               sum([loss_mano_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_mano_params])
            #    self.cfg.LOSS_WEIGHTS['INTERHANDJLOSS'] * loss_interhand_j_loss+\

        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach())

        for k, v in loss_mano_params.items():
            losses['loss_' + k] = v.detach()

        output['losses'] = losses
        # print(f'losses:{losses}')

        return loss

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True, write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        #images = 255*images.permute(0, 2, 3, 1).cpu().numpy()

        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)
        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        focal_length = output['focal_length'].detach().reshape(batch_size, 2)
        gt_keypoints_3d = batch['keypoints_3d']
        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)

        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        gt_keypoints_3d = batch['keypoints_3d']
        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)

        # We render the skeletons instead of the full mesh because rendering a lot of meshes will make the training slow.
        #predictions = self.renderer(pred_keypoints_3d[:num_images],
        #                            gt_keypoints_3d[:num_images],
        #                            2 * gt_keypoints_2d[:num_images],
        #                            images=images[:num_images],
        #                            camera_translation=pred_cam_t[:num_images])
        predictions = self.mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                               pred_cam_t[:num_images].cpu().numpy(),
                                                               images[:num_images].cpu().numpy(),
                                                               pred_keypoints_2d[:num_images].cpu().numpy(),
                                                               gt_keypoints_2d[:num_images].cpu().numpy(),
                                                               focal_length=focal_length[:num_images].cpu().numpy())
        if write_to_summary_writer:
            summary_writer.add_image('%s/predictions' % mode, predictions, step_count)

        return predictions

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step_discriminator(self, batch: Dict,
                                    hand_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            hand_pose (torch.Tensor): Regressed hand pose from current step
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        batch_size = hand_pose.shape[0]
        gt_hand_pose = batch['hand_pose']
        gt_betas = batch['betas']
        gt_rotmat = aa_to_rotmat(gt_hand_pose.view(-1,3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(hand_pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = joint_batch['img']
        mocap_batch = joint_batch['mocap']
        optimizer = self.optimizers(use_pl_optimizer=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            optimizer, optimizer_disc = optimizer

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_mano_params = output['pred_mano_params']
        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)
        loss = self.compute_loss(batch, output, train=True)
        # print(f'loss:{loss}')
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            disc_out = self.discriminator(pred_mano_params['hand_pose'].reshape(batch_size, -1), pred_mano_params['betas'].reshape(batch_size, -1))
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv

        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        optimizer.step()
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            loss_disc = self.training_step_discriminator(mocap_batch, pred_mano_params['hand_pose'].reshape(batch_size, -1), pred_mano_params['betas'].reshape(batch_size, -1), optimizer_disc)
            output['losses']['loss_gen'] = loss_adv
            output['losses']['loss_disc'] = loss_disc

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        # batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output
