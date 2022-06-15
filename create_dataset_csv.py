import os
import pandas as pd
from PIL import Image


def verify_jpeg_image(file_path):
    try:
        img = Image.open(file_path)
        img = img.convert('RGB')
        img.getdata()[0]
        img.close()
    except OSError:
        return False
    return True


def create_dataframe(cat_dir: str, dog_dir: str, out_folder: str):
    cat_images = os.listdir(cat_dir)
    dog_images = os.listdir(dog_dir)

    cat_images = [os.path.join(cat_dir, x) for x in cat_images]
    dog_images = [os.path.join(dog_dir, x) for x in dog_images]

    dog_images_train = dog_images[:11500]
    dog_images_val = dog_images[11500: 12000]
    dog_images_test = dog_images[12000:]

    cat_images_train = cat_images[:11500]
    cat_images_val = cat_images[11500: 12000]
    cat_images_test = cat_images[12000:]

    dog_labels_train = [1 for _ in range(len(dog_images_train))]
    dog_labels_val = [1 for _ in range(len(dog_images_val))]
    dog_labels_test = [1 for _ in range(len(dog_images_test))]

    cat_labels_train = [0 for _ in range(len(cat_images_train))]
    cat_labels_val = [0 for _ in range(len(cat_images_val))]
    cat_labels_test = [0 for _ in range(len(cat_images_test))]

    train_x = dog_images_train + cat_images_train
    val_x = dog_images_val + cat_images_val
    test_x = dog_images_test + cat_images_test

    train_y = dog_labels_train + cat_labels_train
    val_y = dog_labels_val + cat_labels_val
    test_y = dog_labels_test + cat_labels_test

    d_train = {"image_path": train_x, "class": train_y}
    d_val = {"image_path": val_x, "class": val_y}
    d_test = {"image_path": test_x, "class": test_y}

    train_df = pd.DataFrame(data=d_train)
    train_df = train_df.sample(
        frac=1,
        random_state=1
    ).reset_index()

    val_df = pd.DataFrame(data=d_val)
    val_df = val_df.sample(
        frac=1,
        random_state=1
    ).reset_index()

    test_df = pd.DataFrame(data=d_test)
    test_df = test_df.sample(
        frac=1,
        random_state=1
    ).reset_index()

    train_df.to_csv(os.path.join(out_folder, "train"))
    val_df.to_csv(os.path.join(out_folder, "val"))
    test_df.to_csv(os.path.join(out_folder, "test"))


if __name__ == "__main__":
    create_dataframe(
        r"D:\\cnn_vs_transformers\\data\\dogs-vs-cats\\Cat",
        r"D:\\cnn_vs_transformers\\data\\dogs-vs-cats\\Dog",
        r"D:\\cnn_vs_transformers\\data")
