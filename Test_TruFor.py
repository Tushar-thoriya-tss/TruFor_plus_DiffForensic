import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

original_dir = r"/work/sharedrive/gpu3/dev_notebooks/Models/Tamper Detection/20260109_apply_filters_to_images/SBIG_360000_plus images/AadhaarRegular/AadhaarRegular"
output_dir   = r"/work/sharedrive/Tushar Thoriya/GitHub techniques/TruFor/SBIG Data/output_images_eph_100/AadhaarRegular"
output_img_dir = r"/work/sharedrive/Tushar Thoriya/GitHub techniques/TruFor/SBIG Data/filtered_results_eph_100_thr_0_05"

os.makedirs(output_img_dir, exist_ok=True)

npz_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".npz")])

print("Total files found:", len(npz_files))


for file in tqdm(npz_files):
    
    image_name = file.replace(".npz", "")
    img_path = os.path.join(original_dir, image_name)

    if not os.path.exists(img_path):
        print("Image not found:", image_name)
        continue

    npz_path = os.path.join(output_dir, file)
    data = np.load(npz_path)
    
    score = float(data['score']) * 100
    score = round(score, 2)
    
    if float(data['score']) <= 0.05:
        continue

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    map_array = data["map"]

    if len(map_array.shape) == 3:
        map_array = map_array.squeeze()

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(map_array, cmap="jet")
    plt.colorbar()
    plt.title(f"TruFor Localize map: {score}%")
    plt.axis("off")

    plt.tight_layout()

    save_path = os.path.join(output_img_dir, image_name)
    plt.savefig(save_path)
    plt.close()

print("Done.")
