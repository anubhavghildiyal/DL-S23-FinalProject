# Image classification on extrusion dataset

## Introduction
This project presents our approach to detecting under extrusion errors in 3D printing using Vision Transformers, ResNet50, and BEiT models. Under extrusion is a critical issue in 3D printing, characterized by inadequate plastic flow through the nozzle, resulting in various defects. Through extensive experimentation on a comprehensive dataset, we evaluate the performance of these models in accurately classifying 3D printing images for under extrusion. Our findings contribute to improving quality control in 3D printing processes and advancing computer vision techniques in error detection.

## Models
Here we have worked on 3 models, in increasing order of trainable parameters and  model complexity: 
1. ResNet 152 
2. Vision Transformer 
3. BERT Pre-Training of Image Transformers 

### Model 1: ResNet152 
We implemented the ResNet-152 architecture with custom modifications to suit our classification task. The backbone of the network was initialized with a pretrained ResNet-152 model. This pre-trained model was obtained from Microsoft Research, which was originally trained on the ImageNet dataset. The pre-trained model is known for its excellent performance on various image classification tasks. To adapt the ResNet-152 model for our specific task, we adjusted the fully connected layers and the frozen/unfrozen layers. We kept the first three residual blocks frozen, meaning that their weights were not updated during training. This decision was made based on the observation that the initial layers of a deep neural network often capture low-level features that are generally applicable across different tasks. The last residual block (layer 4) was left unfrozen, allowing its weights to be updated during training. This decision was made to enable the network to fine-tune the higher-level features that are more specific to our classification task. By updating the weights in the last residual block, the network can adapt to the unique patterns and features present in our dataset. In addition to the pre-existing layers, we added two fully connected layers followed by ReLU activations, and a final fully connected layer followed by a sigmoid activation. These additional layers were appended after the backbone layers of ResNet-152. The first fully connected layer had 1000 input units, which matched the output size of the ResNet-152 backbone. This layer was followed by a ReLU activation to introduce non-linearity. The second fully connected layer had 1024 units, followed by another ReLU activation. Finally, the last fully connected layer had the number of output units corresponding to the number of classes in our classification task, and it was followed by a sigmoid activation. This sigmoid activation provided the probabilities for each class, allowing for multi-label classification. **Trainable Parameters: 8,062, 442**

### Model 2: Vision Transformer (vit_small_patch16_224)
Our implementation of the Vision Transformer involved custom modifications to adapt it to our specific classification task. The Vision Transformer is a transformer-based architecture that has gained attention for its effectiveness in computer vision tasks. In our implementation, we focused on modifying the classification head of the Vision Transformer (vit_small_patch16_224) . The backbone of the model, consisting of transformer layers, remained unchanged. However, we adjusted the fully connected layers in the classification head. Specifically, we kept the last layer of the Vision Transformer unfrozen, allowing its weights to be updated during training. This decision was based on the understanding that the last layer captures higher-level features that are more specific to our classification task. By fine-tuning this layer, the network can adapt and specialize the learned representations to our dataset. In addition to the pre-existing layers, we introduced two fully connected layers with ReLU activations and a final fully connected layer with a sigmoid activation. These layers were appended after the transformer layers of the Vision Transformer. The first fully connected layer had the appropriate input size to match the output size of the transformer layers. We followed this layer with a ReLU activation, introducing non-linearity to capture complex feature interactions. The second fully connected layer had the desired number of units tailored to our specific classification task. Another ReLU activation was applied after this layer to enhance the expressive power of the network. Finally, the last fully connected layer had the number of output units corresponding to the number of classes in our classification task. We used a sigmoid activation function on this layer to obtain class probabilities, enabling multi-label classification. **Trainable Parameters: 8,062,058** 

### Model 3: BeIT (beit-base-patch16224-pt22k)
Our implementation involved adapting the BEiT (BERT pretraining for Image Transformers) model to suit our classification task. The BEiT model combines the power of BERT (Bidirectional Encoder Representations from Transformers) pretraining with image transformers for improved performance in computer vision tasks. In our implementation, we focused on modifying the classification head of the BEiT model (beit-base-patch16224-pt22k). The backbone of the model, which consists of a stack of transformer layers, remained unchanged. However, we adjusted the fully connected layers in the classification head. Specifically, we added two fully connected layers with ReLU activations and a final fully connected layer with a sigmoid activation. These additional layers were appended after the transformer layers of the BEiT model. The first fully connected layer had the appropriate input size to match the output size of the transformer layers. This layer was followed by a ReLU activation, introducing non-linearity to capture complex relationships between features. The second fully connected layer had the desired number of units tailored to our specific classification task. Another ReLU activation was applied after this layer to enhance the expressive power of the network. Finally, the last fully connected layer had the number of output units corresponding to the number of classes in our classification task. We used a sigmoid activation function on this layer to obtain class probabilities, enabling multi-label classification. **Trainable Parameters: 7,318,400** 

We trained and tested 3 different models to detect “under extrusion” in 3D printer images.  
1. For BEiT,  
  - BeIT Test Accuracy: 0.9930298544288182
  - BeIT Precision: 0.9930317440646658 
  - BeIT Recall: 0.9930298544288182 
  - BeIT F1 Score: 0.9930303374898138  
 ![WhatsApp Image 2023-05-16 at 22 56 53](https://github.com/anubhavghildiyal/DL-S23-FinalProject/assets/115520834/9b4219d6-b6ca-48ec-88de-a67f76139cb6)
![WhatsApp Image 2023-05-16 at 22 57 29](https://github.com/anubhavghildiyal/DL-S23-FinalProject/assets/115520834/c38f7733-1487-4dfd-bf01-59183ce751e6)

 2. For Vision Transformer,
  - ViT Test Accuracy: 0.9781643227239082 
  - ViT Precision: 0.9782257221021233 
  - ViT Recall: 0.9781643227239082 
  - ViT F1 Score: 0.9781477315158604  
  ![WhatsApp Image 2023-05-16 at 23 48 46](https://github.com/anubhavghildiyal/DL-S23-FinalProject/assets/115520834/db8e2f8f-7ce5-4d54-8ea2-e92d32fefb7b)
![WhatsApp Image 2023-05-16 at 23 49 05](https://github.com/anubhavghildiyal/DL-S23-FinalProject/assets/115520834/ffd496d3-f9a1-4594-b681-e210f14f6091)

 3. For ResNet-152,
  -  ResNet-152 Test Accuracy: 0.9686035035775968 
  -  ResNet-152 Precision: 0.9697577813197632 
  -  ResNet-152 Recall: 0.9686035035775968 
  -  ResNet-152 F1 Score: 0.9686609739417156 
![WhatsApp Image 2023-05-16 at 23 58 19](https://github.com/anubhavghildiyal/DL-S23-FinalProject/assets/115520834/448dc8ce-c3e2-4a8a-88d9-cbdd9a244cfc)
![WhatsApp Image 2023-05-16 at 23 58 35](https://github.com/anubhavghildiyal/DL-S23-FinalProject/assets/115520834/f468abab-6e20-4358-a82b-cd8bd9205fb0)

## Conclusion
Based on the evaluation results, it can be concluded that the "BEIT" (Bidirectional Encoder representation from Image Transformers) model outperforms other models in the image classification task by achieving an accuracy of staggering 99.3%. The superior performance of large pre-trained models based on the attention mechanism underscores their effectiveness in computer vision tasks. This represents a notable milestone in the field of deep learning, as these attentionbased models achieve state-of-the-art accuracy compared to traditional convolutional neural networks.            The success of pre-training these large attention models can be attributed to their ability to capture intricate visual patterns and relationships within images. The attention mechanism allows the model to focus on relevant image regions and exploit contextual information effectively. By leveraging massive amounts of pre-training data, these models develop robust visual representations, enabling them to generalize well to downstream tasks like image classification. The advancements in attention-based models signify a paradigm shift in computer vision, offering improved accuracy and performance across various applications.
![WhatsApp Image 2023-05-17 at 00 10 38](https://github.com/anubhavghildiyal/DL-S23-FinalProject/assets/115520834/f34a63a1-2c41-40b0-a5a9-20d87cc8059c)
