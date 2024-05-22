import torch
from torchvision.transforms import ToTensor, Resize, transforms
from PIL import Image
from utils import FeedForwardNet, class_mapping

def predict_image_class(image_path, model):
    # Load the image
    image = Image.open(image_path).convert('RGB') 

    transform = transforms.Compose([
        Resize((640, 480)), 
        ToTensor()
    ])
    image = transform(image)

    image = image.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        predicted_index = torch.argmax(outputs, dim=1).item()

    predicted_class = class_mapping[predicted_index]
    
    return predicted_class

if __name__ == "__main__":

    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("baby.pth")
    feed_forward_net.load_state_dict(state_dict)

    image_path = "output/discomfort/9CFD61B9-BF13-406D-8B2F-F73CFAAF25CB-1430927728-1.0-f-04-hu.png"
    predicted_class = predict_image_class(image_path, feed_forward_net)
    print(f"The predicted class for the image is: {predicted_class}")
