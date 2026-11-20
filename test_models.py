
from modelss.unet import build_unet

from modelss.pix2pix import (
    build_generator,
    build_discriminator
)


# Build U-Net
unet = build_unet()

# Build Pix2Pix models
generator = build_generator()
discriminator = build_discriminator()


print("All models built successfully!")

print("\nU-Net output shape:")
print(unet.output_shape)

print("\nGenerator output shape:")
print(generator.output_shape)

print("\nDiscriminator output shape:")
print(discriminator.output_shape)
