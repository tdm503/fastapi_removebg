import gradio as gr
from PIL import Image

def removebackground(image, mode):
    if mode == "DIS":
        # Xử lý hình ảnh ở đây và trả về ảnh đã được xử lý (sử dụng PIL.Image)
        return "Background removed successfully", image  # Giả sử bạn trả về ảnh gốc hoặc ảnh đã xử lý
    else:
        return "Only DIS is supported in this demo.", None

demo = gr.Interface(
    fn=removebackground,
    inputs=[
        gr.Image(type="pil", label="Input Image"),  # Dùng type="pil" để nhận ảnh dưới dạng đối tượng PIL
        gr.Radio(["DIS"], label="Select Model")    # Thêm lựa chọn kiểu mô hình
    ],
    outputs=[
        gr.Text(label="Status"),                   # Trả về trạng thái
        gr.Image(label="Output Image")             # Trả về ảnh đã xử lý
    ],
    title="Remove Background",
    description="This is a background removal application using the DIS model.",
)

demo.launch()
