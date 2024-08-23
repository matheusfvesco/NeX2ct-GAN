from torchvision import transforms, utils
from pathlib import Path
from PIL import Image
import argparse


def main(args):
    # Base directory containing the subdirectories and CSV file
    INPUT_DIR = Path(args.input)

    print(INPUT_DIR)

    # Directory to save the grids
    grids_dir = INPUT_DIR / "grids"
    grids_dir.mkdir(exist_ok=True)

    # Collect subdirectories
    subdirs = sorted(
        [
            d
            for d in INPUT_DIR.iterdir()
            if d.is_dir() and d.name != "grids" and d.name != "xrays"
        ]
    )

    # writes the labels for the
    with open(grids_dir / "1-grid-labels.txt", "w") as file:
        file.write(", ".join([str(subdir.name) for subdir in subdirs]))

    # Load images from each subdirectory
    def load_images_from_subdir(subdir, image_name):
        image_path = subdir / image_name
        image = Image.open(image_path)  # default loader for image
        return image

    # Finds all the filenames inside the first directory and look for the
    # same filename...
    for image_file in subdirs[0].iterdir():
        if image_file.suffix.lower() == ".jpeg":
            images = []
            for subdir in subdirs:
                img = load_images_from_subdir(subdir, image_file.name)
                transform = transforms.Compose(
                    [
                        transforms.Resize((200, 200)),  # Resize to uniform size
                        transforms.ToTensor(),
                    ]
                )
                img = transform(img)
                images.append(img)

            # Create a grid of images
            image_grid = utils.make_grid(images, nrow=len(subdirs))

            # Save the grid
            grid_path = grids_dir / image_file.name
            utils.save_image(image_grid, grid_path)

            print(f"Saved grid image for {image_file.name} at {grid_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Generates the preprocessing
                            demo strips given the path to where each
                            of the original images were generated"""
    )
    parser.add_argument("input", type=str, help="Input file path.")
    args = parser.parse_args()
    main(args=args)
