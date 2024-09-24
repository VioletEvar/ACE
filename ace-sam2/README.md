## Get amodel mask with SAM2

First, clone the `segment-anything-2` repository and complete the installation of `sam2`:

```bash
git clone git@github.com:facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

Place the `get_amodel_masks.py` file in the `/segment-anything-2/` directory.

Then, prepare the video frame sequences and initial masks:

- Process the videos to extract the frames, and store the video frame sequences in JPEG format.
- Annotate the left hand, right hand, and objects in the first frame of each video to generate the initial masks, and store them in PNG format.
- The final file structure should look like this:

```
videos/
├── frame_sequnces/
│   ├── video_0/
│   │   ├── 0000.jpg
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   ├── ...
│   │
│   ├── video_1/
│   │   ├── ...
│   │
│   ├── ...
│
├── initial_masks/
│   ├── video_0/
│   │   ├── 0/
│   │   │   └── 0000.png
│   │   ├── 1/
│   │   │   └── 0000.png
│   │   └── 2/
│   │       └── 0000.png
│   │
│   ├── video_1/
│   │   ├── ...
│   │
│   ├── ...
│
├── amodel_masks/
```

Finally, run the following command:

```bash
python get_amodel_masks.py --base_video_dir /videos/frame_sequnces/ --input_mask_dir /videos/initial_masks/ --output_mask_dir /videos/amodel_masks/
```

This will generate the separate masks for the left hand, right hand, and objects of each video in the `/videos/amodel_masks/` directory.
