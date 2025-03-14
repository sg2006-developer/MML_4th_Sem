import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


im = Image.open(r"C:\Users\Admin\SG_164\Img2.jpg").convert("L")


img_array = np.array(im)


U, S, Vt = np.linalg.svd(img_array, full_matrices=False)


k = 50
S_k = np.diag(S[:k])
U_k = U[:, :k]
Vt_k = Vt[:k, :]


reconstructed_img = np.dot(U_k, np.dot(S_k, Vt_k))


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img_array, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')


axes[1].imshow(reconstructed_img, cmap='gray')
axes[1].set_title(f"Compressed Image (k={k})")
axes[1].axis('off')



plt.show()
