# U-Net Notes


## Abstract
- To use the available annotated samples more efficiently to train neural network for segmentation.
- [ ] Initial contraction (Shrinking of image) - to capture context. How?
- [ ] Symmetrical Expansion (of image) - for localisation. How?
- Trained with very small images, but outperforms Sliding window technique.
    - [ ] Look how sliding window segmentation implemented.
- Faster than earlier methods (prior to 2015)

## What I infer from Figure 1
![U Net Architecture](/images/Architecture.png)
- There exist several blocks - where 2 convolution operation done then either downscaling(halving height and width,then doubling channel) or upscaling operation( doubling height and width, halving channel at the same time)
    <!-- - But there is a difference in the way channel size is upscaled and downscaled.
    - In contraction, while maxpooling channel size remain same. Filter size then increased to change channel size.
    - In expansion, while upscaling operation is done the channel dimension is directly changed. and then once again halved by using filters. -->
- On each subsequent block in left, the image size (or width and height) is reduced. But the width of channels increased (by using more number of filters in convolution)
- From the bottom, the input data is upscaled, and at each subsequent blocks the width of channels are subsequently decreased.
- While upscaling features extracted from *symmetrical* blocks are transferred(added) to the currently upscaled volume.
    - So the actual upscaled feature map is the sum of transferred feature map + up-convoled feature map from previous block.
- Upscaling is symmetrical to downscaling but the input dimensions are *not exactly* symmetrical before downscaling and after scaling. 

- Initial input size 572 by 572 image. Final output size 388 by 388. Only **2** output channels(mask)
    - [x] Then how does all pixels are segmented?
        - Refer figure 2. "Missing input data is extrapolated by mirroring" 
        - Since valid padding is used in this paper, the size of feature map decreases but even when upsampling *it didnt have exact size of input*, hence output mask doesn't completely represent input image. So, the missed out border pixels are mirrored(Extrapolated) so that now even with reduced output mask they are also included. 

## What I infer from Figure 2
![Overlap tile strategy](Overlap-tile-/images/strategy.png)
- First image is input image, which has higher dimension (height, width) compared to output mapping.
- Second image is output to be overlapped with input image for segmenting.
- The whole pixels inside blue box is needed for segmenting the pixels in yellow box.
    - [x] Why ?
        - Because of valid padding and even while upscaling the spatial size is not maintained with input image size.
- [x] What is overlap tile strategy?
    - It is like sliding window strategy where large images (whose dimension doesn't fit the model's input dimension) are divided into fixed size overlapping windows/patches. And each patch is used one by one by the model.
    - Optimal tile size need to be selected. Overlapping regions in output should be carefully dealt with.

## What I infer from Figure 3
![Segmented results](/images/fig3.png)
- a. Raw image
- b. Ground truth image for training
- c. Trained output mask, which when overlayed on input image gives the segmentation 
    - I guess Multiple masks are added (# of masks = # of class)
- d. Image of Proposed weight loss method that gives more preference to the bakcground seperating two borders. Dark red represents more weight to thin border. Dark blue represents less weight.


## Introduction
- (Existing Problem 1) - Success of CNN is limited to availabilty if training sets and the size of network. Thousands of training images are usually beyond the reach in biomedical tasks.
    - (Existing Solution 1) - Ciresan et al trained sliding window model with small multiple patches of same image. Localises(Can determine that in this particular pixel dimension there is this class etc) as well as gets to have multiple training data.
- (Existing Problem 1.2) - Ciresan et al's method is very slow. Network to run on each patch and Redundant overlapping patches. Tradeoff (localisation accuracy vs context  tradeoff - refer Reading list)
    - (Solution 1.2) - In this paper, FCN (fully convolutional network) is used to get better from the tradeoff.
- (Problem 2) - For less training data problem
    - (Solution 2) - Excessive data augmentation including *Elastic deformations* done.
- (Problem 3) - Problem of seperation of touching objects while segmenting.
    - (Solution 3) - Proposed weighted loss where background labels in such thin borders are given large weight in loss function.

- Main Contributions in this paper
    - To increase the resolution of output, the feature maps are upscaled.
    - To increase localization, high resolution features from contracting path are supplied(via skip connections) to feature maps being upscaled.
    - Having Large feature channels even after upscaling in each block enhances passage of contextual information to higher resolution layers.


## Network Architecture
- **Left side** (Downsampling/Contracting path)
    - Typical CNN Architecture
    - 3 by 3 convolution of feature maps. Unpadded - so valid padding(Size of feature map decreases)
        - Each followed by ReLU
    - Then 2 by 2 Max pooling with stride 2 - downsampling step.
    - Feature channels are doubled after downsampling step   
- **Right side** (Upsampling/Expanding path)
    - First upscaled (somehow done and seems like channel dim maintained) then 2 by 2 convolution done (up convolution, which halved feature channels dim). (But usually on convolution x,y size changes and channel size depends on filters used)
        - [x] How does `up convolution` decrease channel width? 
            - `Up convolution` in original U Net paper is done with `transposed convolution`. 
            - It is similar to normal convolution but in reverse, thats all. So like increasing/deacreasing number of filters in usual convolution increases/decreases channel depth, the same happens here.
    - After halving, channel size is increased(or maintained from previous block) by concatenating corresponding **cropped** feature map from contraction path.
        - [x] Why cropping needed?
            - Cropping is necessary because the upsampled feature map's spatial size is not exactly equal to its respective contractive path's feature map. Because of valid padding border pixels are lost right? So, we need equal spatial size for skip connection value and upsampled feature map to be concatenated. hence cropping is done.
            - See figure 1 for more claity.
    - Then 3 by 3 convolution followed by ReLU.
- Finally 1 by 1 convolution done (i.e filter size is 1 by 1 by 64 here). 2 classes, so two filter used.
- Total of 23 convolutions.
    - 8 **after** downsampling
    - 2 at bottom of architecture
    - 4 **while** upsampling
    - 8+1 = 9 **after** upsampling

## Training
- Stochastic Gradient used with Caffe.
> Remember: less the batch size, slower it takes to converge. But possibly reach a good minima with slow updation.
-  Here batch size is taken as 1 to reduce computational overhead. The authors favour **large input tiles** 
    - [ ] (Why computer overhead reduced and why large input tile preferred?)
- High momentum (0.99) used,
    - [ ] Revisit Momentum's mathematical concept.
- Final layer used  SoftMax along with Cross entopy loss function.
    - [ ] Revisit mathematics of Cross Entropy and its benifits.
- Weight map is precomputed to compensate different frequencies of classes(border, target, background)
    - Given different weight to different classes. As per my understanding, more weight is given to thin border, followed by target classes and least weight/preference to backgrounds.
- Effective (random) Initilialisation of weights is very important to give equal preference to all parts of the network.   
    - [ ] Each feature map(weights?) Should have unit variance, why?
    - drawn from gaussian distribution of sqrt(2/N). N - filter size * number of feature channels. if 3 by 3 filer, 3 channel, then N = 9 * 3 = 27.
        - [ ] Understand the intuition
- [ ] How is the weight map used in training?
    
## Data Augmentation
- Performed to improve invariance(Model predicting proper resuts even after change in orientation, scaling, lighting of input image) and robustness(Accurate results even in presence of noisy or corrupted data).
- Specific to the application domain. Here biological cells based invariance and robustness is given importance needed. 
    - Invariance & variance to : shift, rotation,  deformations and gray value variations(change in intensity or brightness).
    - Especially **random elastic deformations**
        - [ ] Read again about how it is done.
- [ ] How does **Dropout layers** at end of contracting path perform implicit data augmentation?

## Experiments
- Experimented with three types of segmentation tasks
    1. segmentation of neuronal structures(30, 512 by 512 images).
    1. segmentation of Glioblastoma-astrocytoma U373 cells (35 partially annotated data).
    1. segmentation of HeLa cells(20 partially annotated data).
- [x] Why in first task IoU metric not used whereas in second and third task it is used?
    - Used for the sake of competition evaluation. Even IoU is popular, but to get a more wide understanding of model's performance in different aspects different metrics are used.
    - [ ] Intuitively Understand the importance of Rand, Pixel and warping error.
- Trained for **10 hours** in Nvidia Titan GPU(6 GB)

- Results for task 1. 
![task 1](/images/Exp1.png)

- Results for task 2 and 3 with IoU
![task 2 and 3](/images/Exp2_3.png)

## Other Reading List
- [ ] What did *Krizhevsky et al* done as a break through ? (Mentioned in Introduction)
- Trade-off in  contribution of *Ciresan et al* 
    - [x] Why does using max pooling on larger patches  affects localization accuracy? *(Before, read about spatial resolution)*
        - Reducing spatial size by max pooling doesn't reduce spatial resolution (number of pixels per inch, PPI) but reduces spatial size.
        - Example: if initial spatial size is 1000 by 1000(resolution of 100 PPI), after max pooling of 2 by 2 frame, spatial size is 500 by 500. But the resolution remains same, i.e 100 PPI.
            - Now 1 pixel represents the info of 2 pixel. hence spatial information is lost. Which in turn affects localisation of objects, as the pixel detail of object is lost somehow.

    - [x] Why does small patches allow  little context.
        - Context refers to information in the image beyond the object like background that helps in predictions. With very small pixel region you get low context.
        - Also, Small patch has more localisation because if the patch size decreases to very small, then ideally pixels of only one class would be present.
- [x] Why does upsampling increases resolution of image?
    - Upsampling increases spatial size, spatial size increases number of pixels. Now more pixel represent the same information earlier represented my less number of pixels. hence resolution is increased.
- [x] Spatial size vs Input size
    - Spatial size is also height and width but it refers to the feature maps (or activations) within the network which decreases upon convolution and pooling operation. But input size refers to input data size that is constant unless we resize it.
- [x] Spatial size vs resolution
    - Spatial size - refer to physical dimension
    - Spatial resolution - refer to pixel dimension, i.e detail present in image. Consider a 10 inch by 10 inch image with spatial resolution of 100. That means 100 pixels per inch, total of 1000 by 1000 pixes. 
        - So higher the pixel per inch, higher is the spatial resolution higher is the image details.
- [x] Why using Valid padding(Though inconvinient) instead of same padding.
    - Because of border effect of including lot of zeros in padding which reduces performance.
    - [ ] Why don't they try making the output mask same as size of input image even with Valid padding? Is it a performance concern?