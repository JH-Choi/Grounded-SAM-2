"""Extract masks from images using Grounded SAM2."""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# Default model paths
DEFAULT_SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_GROUNDING_DINO_ID = "IDEA-Research/grounding-dino-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract masks using Grounded SAM2")
    parser.add_argument("--text_prompt", type=str, required=True, help="Text prompt for object detection")
    parser.add_argument("--image_path", type=str, help="Path to a single image file")
    parser.add_argument("--image_dir", type=str, help="Path to directory of images")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", help="Output directory for masks")
    parser.add_argument("--vis_dir", type=str, default="vis_masks", help="Output directory for visualizations")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="Box threshold for detection")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text threshold for detection")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualizations")
    parser.add_argument("--sam2_checkpoint", type=str, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--sam2_config", type=str, default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--grounding_dino_id", type=str, default=DEFAULT_GROUNDING_DINO_ID)
    return parser.parse_args()


def setup_autocast(device: str) -> None:
    """Configure autocast and TensorFloat-32 settings for GPU."""
    if device == "cuda":
        torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def save_visualization(
    image: Image.Image,
    mask: torch.Tensor,
    output_path: str,
) -> None:
    """Save side-by-side visualization of image and mask."""
    image_array = np.array(image.convert("RGB"))
    h, w = image_array.shape[:2]

    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255
    mask_tensor = mask.float().cpu()

    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0)

    resized_mask = F.interpolate(
        mask_tensor.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
    ).squeeze(0)
    resized_mask = resized_mask.repeat(3, 1, 1)

    concat_img = torch.cat([image_tensor, resized_mask], dim=2)
    torchvision.utils.save_image(concat_img, output_path)


def extract_mask(
    image_path: str,
    text_prompt: str,
    sam2_predictor: SAM2ImagePredictor,
    grounding_model,
    processor: AutoProcessor,
    device: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    vis_dir: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Extract object masks from an image using Grounded SAM2.

    Args:
        image_path: Path to the input image.
        text_prompt: Text prompt for object detection.
        sam2_predictor: Initialized SAM2 image predictor.
        grounding_model: Initialized Grounding DINO model.
        processor: Grounding DINO processor.
        device: Device to run inference on.
        box_threshold: Confidence threshold for bounding boxes.
        text_threshold: Confidence threshold for text matching.
        vis_dir: Directory to save visualizations (None to skip).

    Returns:
        Extracted mask as numpy array, or None if no objects detected.
    """
    image = Image.open(image_path).convert("RGB")
    base_name = Path(image_path).stem

    # Run Grounding DINO
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )

    input_boxes = results[0]["boxes"]
    labels = results[0]["labels"]

    if len(input_boxes) == 0:
        print(f"No objects detected in {image_path}")
        return None

    print(f"Detected {len(labels)} objects: {labels}")

    # Run SAM2
    sam2_predictor.set_image(np.array(image))
    setup_autocast(device)

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # Combine all masks into single binary mask
    combined_mask = np.any(masks, axis=0).astype(np.uint8)
    mask_tensor = torch.from_numpy(combined_mask).to(device)

    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0)

    # Save visualization if requested
    if vis_dir is not None:
        vis_path = os.path.join(vis_dir, f"{base_name}_{text_prompt.strip()}.jpg")
        save_visualization(image, mask_tensor[0], vis_path)

    return mask_tensor.cpu().numpy()


def load_models(args: argparse.Namespace):
    """Load SAM2 and Grounding DINO models."""
    print("Loading SAM2...")
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=args.device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    print("Loading Grounding DINO...")
    processor = AutoProcessor.from_pretrained(args.grounding_dino_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.grounding_dino_id
    ).to(args.device)

    return sam2_predictor, grounding_model, processor


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.debug:
        os.makedirs(args.vis_dir, exist_ok=True)

    sam2_predictor, grounding_model, processor = load_models(args)

    # Collect image paths
    image_paths = []
    if args.image_path:
        image_paths.append(args.image_path)
    elif args.image_dir:
        for filename in sorted(os.listdir(args.image_dir)):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(args.image_dir, filename))
    else:
        print("Error: Either --image_path or --image_dir must be provided")
        return

    # Process images
    for image_path in tqdm(image_paths, desc="Processing images"):
        mask = extract_mask(
            image_path=image_path,
            text_prompt=args.text_prompt,
            sam2_predictor=sam2_predictor,
            grounding_model=grounding_model,
            processor=processor,
            device=args.device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            vis_dir=args.vis_dir if args.debug else None,
        )

        if mask is not None:
            base_name = Path(image_path).stem
            output_path = os.path.join(args.output_dir, f"{base_name}.npy")
            np.save(output_path, mask)
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
