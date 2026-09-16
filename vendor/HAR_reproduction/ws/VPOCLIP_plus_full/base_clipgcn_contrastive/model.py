"""
CLIP-COCO contrastive learning scaffold.

Target architecture:
    image encoder: written and trained by you
    text encoder : a pretrained OpenAI CLIP text encoder with frozen parameters

The key steps in this file are commented in detail so you can follow and modify them.
During training you typically only need to:
    1. Prepare a DataLoader whose batches return images and captions
    2. Call model(images, captions)
    3. Apply cross-entropy to logits_per_image / logits_per_text
    4. Let the optimizer update only image_encoder and logit_scale
"""

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# 1. Import the CLIP package
# ---------------------------------------------------------------------------
# Your clipgcn environment already has the local CLIP installed:
#   pip install -e /workspace/CLIP
# so it can be imported directly here.
import clip


def l2_normalize(features):
    """Normalize features to unit length so that the dot product equals cosine similarity."""

    return F.normalize(features, dim=-1)



class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(in_channels=num_channels,  out_channels=num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = self.ReLU(y+x)
        return y


class CustomImageEncoder(nn.Module):
    """
    The part you should mainly modify: write your own image encoder.

    Input:
        images is an image tensor of shape [batch_size, 3, image_size, image_size]

    Output:
        image_features is the image feature of shape [batch_size, embed_dim]

    Key requirements:
        1. The output dimension must equal embed_dim, the output dimension of the CLIP text encoder.
           For example, ViT-B/32 has embed_dim = 512.
        2. forward returns unnormalized features only; normalization is handled uniformly by the outer ContrastiveModel.
        3. The parameters of this module are trainable, so do not use no_grad here.

    A very small CNN baseline is given below; it runs through the training pipeline.
    You can replace self.backbone / self.projection with your own ResNet, ViT, CNN-GCN, etc.
    """

    def __init__(self, embed_dim):
        super(CustomImageEncoder, self).__init__()

        # Step 1: Build the image feature extraction network.
        # The example below progressively downsamples the RGB image, then squeezes it to [B, C, 1, 1] with AdaptiveAvgPool2d.
        # For real experiments you can replace this entire nn.Sequential with your own network.
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.b2 = nn.Sequential(Residual(64, 64, use_1conv=False, strides=1),
                                Residual(64, 64, use_1conv=False, strides=1))

        self.b3 = nn.Sequential(Residual(64, 128, use_1conv=True, strides=2),
                                Residual(128, 128, use_1conv=False, strides=1))

        self.b4 = nn.Sequential(Residual(128, 256, use_1conv=True, strides=2),
                                Residual(256, 256, use_1conv=False, strides=1))

        self.b5 = nn.Sequential(Residual(256, 512, use_1conv=True, strides=2),
                                Residual(512, 512, use_1conv=False, strides=1))
        
        self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

        # Step 2: Build the projection head that maps the backbone output into the CLIP text feature space.
        # If the final output channels of your backbone are not 256, change the input dimension of the first Linear.
        self.projection = nn.Sequential(
            nn.Flatten(),  # [B, 256, 1, 1] becomes [B, 256]
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim),  # the final output must be [B, embed_dim]
        )

    def forward(self, images):
        # Step 3: Chain the modules together in forward.
        # Note: no softmax, no classification head, and no CrossEntropyLoss here.
        # Contrastive learning only needs an output vector to compute similarity against text_features.
        x = self.b1(images)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        image_features = self.projection(x)
        return image_features


class FrozenCLIPTextEncoder(nn.Module):
    """
    Wrapper around the pretrained CLIP text encoder.

    You only need to know three interfaces:
        clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        tokens = clip.tokenize(["a dog", "a cat"], truncate=True).to(device)
        text_features = clip_model.encode_text(tokens)

    Notes:
        - clip.tokenize turns strings into token ids, shape = [B, 77]
        - encode_text outputs text vectors, shape = [B, embed_dim]
        - All CLIP parameters are frozen here; it serves only as a fixed teacher / target space
    """

    def __init__(self, model_name="ViT-B/32", device=None, download_root=None):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # Step 1: Load CLIP. On the first run, if there are no local weights, they are downloaded to ~/.cache/clip
        # or to the download_root you pass in.
        self.clip_model, _ = clip.load(
            model_name,
            device=device,
            jit=False,
            download_root=download_root,
        )

        # Step 2: Freeze the text encoder. CLIP parameters are not updated during training; only your own image encoder is updated.
        self.clip_model.eval()
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

        # Step 3: Record the output dimension. The image encoder output must match this dimension.
        self.embed_dim = int(self.clip_model.text_projection.shape[1])

    def get_device(self):
        # Return the device where the CLIP text encoder currently resides, e.g. cpu, cuda, cuda:0.
        # This way the latest device is still available here even after model.to("cuda") is called later.
        return next(self.clip_model.parameters()).device

    def forward(self, captions):
        # Step 4: captions can be a single string or a list of strings for one batch.
        # truncate=True prevents errors from an occasional overly long COCO caption.
        with torch.no_grad():
            tokens = clip.tokenize(captions, truncate=True).to(self.get_device())

            # Step 5: Output text_features. Unnormalized features are still returned here; the outer model normalizes uniformly.
            text_features = self.clip_model.encode_text(tokens)
            return text_features.float()


class CLIPCOCOContrastiveModel(nn.Module):
    """
    Outermost contrastive learning model.

    Outputs of forward:
        logits_per_image: shape = [B, B]
        logits_per_text : shape = [B, B]

    The i-th image and the i-th caption form the positive pair.
    All other captions / images in the same batch automatically act as negatives.
    """

    def __init__(self, text_model_name="ViT-B/32", device=None, download_root=None):
        super().__init__()

        self.text_encoder = FrozenCLIPTextEncoder(
            model_name=text_model_name,
            device=device,
            download_root=download_root,
        )
        self.image_encoder = CustomImageEncoder(embed_dim=self.text_encoder.embed_dim)

        # The original CLIP paper commonly uses temperature = 0.07, so logit_scale is initialized to log(1/0.07).
        # This is a trainable parameter; during training the scale of the similarity scores is learned automatically.
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        # Move your own image encoder and logit_scale to the same device as the text encoder.
        # Otherwise errors occur on CUDA machines where images are on GPU but image_encoder weights are still on CPU.
        self.to(self.text_encoder.get_device())

    def train(self, mode=True):
        # Train your own image encoder normally.
        super().train(mode)

        # But the CLIP text encoder is frozen, so it always stays in eval mode.
        self.text_encoder.clip_model.eval()
        return self

    def forward(self, images, captions):
        # Step 1: Make sure the images are on the same device as the text encoder.
        images = images.to(self.text_encoder.get_device())

        # Step 2: Obtain image/text features separately.
        # image_features carry gradients and train your image encoder.
        # text_features have no gradient because the CLIP text encoder is frozen.
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(captions)

        # Step 3: Normalize, then do matrix multiplication to get pairwise similarities within the batch.
        image_features = l2_normalize(image_features)
        text_features = l2_normalize(text_features)

        # Step 4: logit_scale.exp() is equivalent to 1 / temperature.
        # clamp prevents the temperature scale from exploding during training.
        scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

    def contrastive_loss(self, images, captions):
        """
        A minimal training loss example.

        labels = [0, 1, 2, ..., B-1]
        indicates that the i-th image should match the i-th caption.
        """

        logits_per_image, logits_per_text = self(images, captions)
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=logits_per_image.device)

        image_to_text_loss = F.cross_entropy(logits_per_image, labels)
        text_to_image_loss = F.cross_entropy(logits_per_text, labels)
        return (image_to_text_loss + text_to_image_loss) / 2


def build_model(text_model_name="ViT-B/32", device=None, download_root=None):
    """
    Constructor for the training script.

    Example:
        model = build_model(device="cuda")
        loss = model.contrastive_loss(images, captions)
        loss.backward()
        optimizer.step()
    """

    return CLIPCOCOContrastiveModel(
        text_model_name=text_model_name,
        device=device,
        download_root=download_root,
    )


def train_step_example(model, optimizer, images, captions):
    """
    Pseudo-code for a single training step, so you can copy it when writing model_train.py.

    Note:
        captions must correspond one-to-one with images.
        For example, images[0] corresponds to captions[0], images[1] to captions[1].
    """

    model.train()

    # The CLIP text encoder is frozen, but model.train() switches all submodules to train mode.
    # So switch it back to eval here to avoid state changes such as dropout/bn.
    model.text_encoder.clip_model.eval()

    optimizer.zero_grad(set_to_none=True)
    loss = model.contrastive_loss(images, list(captions))
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device=device)

    print("device:", next(model.image_encoder.parameters()).device)

    # torchsummary only handles models whose input is a tensor.
    # The forward of the whole CLIPCOCOContrastiveModel requires two inputs, images and captions,
    # and captions is a list of strings, so do not call summary(model, ...) directly.
    # To see the structure, summarizing your own image_encoder is enough.
    from torchsummary import summary
    summary(model.image_encoder, (3, 224, 224), device=device)

    # # To test the whole contrastive model, both images and text must be provided.
    # # Here a fake batch is constructed only to check whether forward runs.
    # model.eval()
    # images = torch.randn(2, 3, 224, 224)
    # captions = ["a dog on the grass", "a cat on the sofa"]
    # with torch.no_grad():
    #     logits_per_image, logits_per_text = model(images, captions)

    # print("logits_per_image:", logits_per_image.shape)
    # print("logits_per_text:", logits_per_text.shape)
