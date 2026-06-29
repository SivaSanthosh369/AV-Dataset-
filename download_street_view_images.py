import os
import numpy as np
import cv2
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = "carla_simulated_dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

print("Generating Local CARLA-Style Driving Dataset...")
print("Creating perfect Road Images and Segmentation Masks (100% Offline)...\n")

# മോഡൽ ട്രെയിനിംഗിനായി 40 സിമുലേറ്റർ ചിത്രങ്ങൾ ഉണ്ടാക്കുന്നു
for i in tqdm(range(40), desc="Generating Simulator Data"):
    # 1. ബ്ലാങ്ക് ക്യാൻവാസ് ഉണ്ടാക്കുന്നു (HD വൈഡ് റസല്യൂഷൻ: 720x1280)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    mask = np.zeros((720, 1280), dtype=np.uint8) # 1-Channel Mask for Segmentation
    
    # ബാക്ക്ഗ്രൗണ്ട് ആകാശം/പരിസരം (Sky/Environment)
    image[0:360, :] = [135, 206, 235] # നീലാകാശം
    
    # 2. റോഡ് ഉണ്ടാക്കുന്നു (Perspective view - ദൂരേക്ക് പോകുമ്പോൾ ചെറുതാകുന്ന റോഡ്)
    road_pts = np.array([[100, 720], [550, 360], [730, 360], [1180, 720]], np.int32)
    cv2.fillPoly(image, [road_pts], [60, 60, 60]) # ഗ്രേ കളർ റോഡ്
    cv2.fillPoly(mask, [road_pts], 1) # Mask-ൽ റോഡിന്റെ വാല്യൂ = 1 (Road Class)
    
    # 3. ലെയ്ൻ മാർക്കിംഗുകൾ (Lane Lines - വെള്ള വരകൾ)
    # ഇടത് വശത്തെ ലൈൻ
    cv2.line(image, (300, 720), (580, 360), (255, 255, 255), 8)
    cv2.line(mask, (300, 720), (580, 360), 2, 8) # Mask-ൽ ലൈനിന്റെ വാല്യൂ = 2 (Lane Class)
    
    # വലത് വശത്തെ ലൈൻ
    cv2.line(image, (980, 720), (700, 360), (255, 255, 255), 8)
    cv2.line(mask, (980, 720), (700, 360), 2, 8)
    
    # സെന്റർ ഡാഷ്‌ഡ് ലൈൻ (ചെറിയ വ്യത്യാസങ്ങൾ വരുത്താൻ ഇമേജ് ഇൻഡെക്സ് ഉപയോഗിക്കുന്നു)
    for y in range(380, 720, 60):
        if (y + i*5) % 120 < 60:
            # ലൈൻ കോർഡിനേറ്റുകൾ കാൽക്കുലേറ്റ് ചെയ്യുന്നു
            x = int(640 + (y - 360) * 0.1)
            cv2.line(image, (x, y), (x + 2, y + 30), (255, 255, 255), 5)
            cv2.line(mask, (x, y), (x + 2, y + 30), 2, 5)

    # 4. റിയലിസ്റ്റിക് ആകാൻ ചെറിയ വെളിച്ച വ്യത്യാസങ്ങൾ (Noise/Shadows) ചേർക്കുന്നു
    noise = np.random.randint(-15, 15, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # ഫയലുകൾ സേവ് ചെയ്യുന്നു
    cv2.imwrite(os.path.join(IMAGES_DIR, f"frame_{i:03d}.png"), image)
    cv2.imwrite(os.path.join(MASKS_DIR, f"mask_{i:03d}.png"), mask)

print(f"\n🎉 SUCCESS! Synthetic CARLA-style dataset successfully generated.")
print(f" -> Images folder: '{IMAGES_DIR}' (Input Features)")
print(f" -> Masks folder: '{MASKS_DIR}' (Ground Truth Labels)")