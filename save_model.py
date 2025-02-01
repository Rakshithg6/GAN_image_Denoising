import torch

# Define a simple GAN model
class SimpleGAN(torch.nn.Module):
    def __init__(self):
        super(SimpleGAN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# Initialize and save the model
model = SimpleGAN()
torch.save(model.state_dict(), "gan_denoiser.pth")

print("✅ Model saved successfully as 'gan_denoiser.pth'")

