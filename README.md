# 🎨 Neural Style Transfer & Stable Diffusion Fusion

This project explores **art style transfer** using deep learning, combining the power of **VGG-19-based Neural Style Transfer** with **Stable Diffusion** to create visually stunning, stylized images. The final output includes a **user-friendly Streamlit app** for interactive experimentation.

---

## 🚀 Features

- ✅ **Neural Style Transfer** using VGG-19
- 🎨 Support for **4 artistic styles** (Van Gogh, Picasso, Andy Warhol, Leonardo da vinci)
- ⚙️ Optimized using the **Adam optimizer**
- 🤖 Integration of **Stable Diffusion** to generate stylized base images
- 🔁 **Hybrid Approach:** Combining diffusion-based generation with neural style transfer
- 🖥️ **Streamlit Web App** for easy user interaction and real-time results

---

## 📚 Technologies Used

- **Python**
- **PyTorch**
- **VGG-19** (pretrained on ImageNet)
- **Stable Diffusion** 
- **Streamlit** (for the front-end interface)
- **Matplotlib**, **PIL**, **NumPy** for image processing

---

🧠 Methodology
Content + Style Input:
The user selects a content image and one of four pre-defined style images( styles can be blended).

Neural Style Transfer:
Feature extraction is performed using a pretrained VGG-19 model.
Computes both content loss and style loss using Gram matrices.
Stylized image is optimized using the Adam optimizer.

Stable Diffusion (Notebook Only):
In a separate notebook, images are generated using Stable Diffusion from text prompts.
These generated images can optionally be used as content images for style transfer.
This part is not included in the Streamlit app.

Streamlit Interface:
Simple UI for uploading content images and selecting styles.
Stylized output is shown in real time.



