# Variational Autoencoder for chess X-ray images with pneumonia

The aim of this project was to learn how to use Variational Autoencoders (VAE) to reduce dimensionaly representation of an image data and how to use VAE as a synthetic data generator. I used chest X-ray dataset with pacients who were healty and those who suffer from pneumonia.

## Image encoding and decoding

The initial goal was to create VAE to make available reducing images from their initial size to a latent space and be able to efectivly decode them from that latent space to an image. I did that by first resizing all the images to a size of 512x512 pixels with Bilinear interpolation in case of a smaller images. It has to be done so all the tensors were the same size. After that I run a handwritten VAE consisting of encoder, decoder with added reparameterization trick, so the autoencoder become variational and to enable gradient propagation.

Example of resized real image:<br />
![image](https://github.com/user-attachments/assets/97f6e5ca-783c-4f50-9a41-5872341fe362)

Example of the same image after encoding and decoding the data after 1st epoch:<br />
![image](https://github.com/user-attachments/assets/bbb42489-313c-4d18-89d9-e00af88a6b6f)

... and after 50th epoch:<br />
![image](https://github.com/user-attachments/assets/d2ed675f-12a2-45fe-8a63-9d8fcc6c1ae1)

As u can see, the initial autoencoder creates very blury image which is sharpened at the end of a training stage.

The loss function used in training process is a combination of regular binary cross-entropy to address the data reconstruction and KL divergence to make latent space a normal distribution so it can work as a synthetic data generator.

## Synthetic data generator

The one of the biggest advantages of VAE over regular autoencoder is that it can work as a data generator cause it has density function of normal distribution due to a KL divergence part in loss function. Below I show how VAE generates an image from a random noise of image shape generated from gaussian distribution.<br />
![image](https://github.com/user-attachments/assets/6625be38-1460-447b-87db-d672b59f3f53)

It is worth noting that it should be compared with decoded real images and not with real images themselves. It surly would have resamble real images better if I had a better GPU to train VAE on.

## Classification and component analysis

After I've trained a VAE I wanted to check how well it extracts features from a real images. I run t-SNE on real images and encoded images to check if visually it creates better base for component analysis. In my opinion it does.

t-SNE on real images:<br />
![image](https://github.com/user-attachments/assets/f18f68d0-07d1-4caa-9edb-82b3ea4eed29)

t-SNE on encoded images:<br />
![image](https://github.com/user-attachments/assets/debc03e8-4317-49fe-a22e-b612bfef8c8a)

Finally I tested how much information is preserved in the encoded image. To do that I run the Support Vector Classification (SVC) on real images, encoded images and decoded images. The results imply very strong ability by VAE to extract features as the SVC on real images got 92% acc (same as decoded images) and 88% on encoded images! That is a huge success because the encoded image has only 400 features in total compared to 512 x 512 features that has a real, resize image.

### Grid of 25 real and decoded images
![image](https://github.com/user-attachments/assets/79396d17-a564-4aea-93e8-9de7d7246681)
![image](https://github.com/user-attachments/assets/3cbc8aaf-5087-4c7a-aafb-e173c3de2c32)



Dataset is taken from https://data.mendeley.com/datasets/rscbjbr9sj/3.
 
