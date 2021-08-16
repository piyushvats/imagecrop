Document Image Tilt and Crop Utility

Main features for this script : 
- Auto rotate for image if it is tilted. 
- Crop words from image (handwriten or printed)for create word image dataset.

Modify input image path and output dir and use.

We can use thid for create image data set of words in documents images i.e invoice, shop payment slip
,online and offine prescription.


For example 
![example1](https://user-images.githubusercontent.com/43409588/129538872-50038555-f5b5-450d-bcc0-5cd129ffd6f2.png)

crop detected text area into 

![exm1_7](https://user-images.githubusercontent.com/43409588/129539509-dab90149-6d61-4be4-970f-7c2feffba837.png)
![exm1_6](https://user-images.githubusercontent.com/43409588/129539515-8d7cc7bd-29e4-45ad-8827-69dd10d93e7e.png)
![exm1_5](https://user-images.githubusercontent.com/43409588/129539517-6a267d41-d22b-4d0b-bb06-710b9a26fbca.png)
![exm1_4](https://user-images.githubusercontent.com/43409588/129539520-b566ce7a-3baf-487f-9561-2ddcb0e67e54.png)
![exm1_3](https://user-images.githubusercontent.com/43409588/129539522-d91bc62c-ccbf-42d1-92fe-2c26c1f31069.png)
![exm1_2](https://user-images.githubusercontent.com/43409588/129539523-eece49c8-27d6-4b3d-9794-00e344501379.png)
![exm1_1](https://user-images.githubusercontent.com/43409588/129539525-97b76489-cb75-439c-9e17-236feb84ffe0.png)

Similarly we used this on other computer generated bill or invoice image

For tilt , we tried histogram method for pixel density but it does not give good result
We used tilt using Houghline method for detect radius and angle , which we used for generate angle from base. This menthod is also not 100% in all scenario 
but batter than all other.


Requirements:- 

-numpy

-matplotlib

-statistics

-opencv


Licence:
MIT
