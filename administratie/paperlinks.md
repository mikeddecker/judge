# Paperlinks, models...

Possible usage of [Jumphabet](https://www.natekg.com/wp-content/uploads/2020/07/Jumphabet-File.pdf)

RepNet: https://sites.google.com/view/repnet (we use a modified version of this for NextJump, could be used for trick segmentation or multiple under counting)

NextJump online demo: https://huggingface.co/spaces/dylanplummer/NextJump (demo showing some basic "jump type" detection e.g double under, triple unders, jogging step)

DensePose for multi-person segmentation from videos: http://densepose.org/ (we have tested this as a backbone for freestyle representations)

Sam2 for prompted video segmentation: https://ai.meta.com/sam2/ (could be useful for labeling jumpers and ropes from videos)

## har sports

HAR in team sports survey (Yin H. 2024-9) [springer](https://link.springer.com/article/10.1007/s10462-024-10934-9)

> The transformer approach would consume a tremendous amount of calculation resources during the training process, and the video itself contains a large amount of redundant information. Moreover, the transformer also highly relies on the extra dataset to achieve good performance. Video masked autoencoders (**VideoMAE**) (Tong et al. 2022) proposed a highly efficient masked autoencoder, which aimed to solve the problem of training transformer models without additional pre-training weights and data. It utilised plain ViT backbones and a high mask ratio to lower the dimensions of inputs, which makes it able to feed into the spatio-temporal self-attention model without requiring a large amount of calculation. Video masked autoencoder v2 (**VideoMAE v2**) (Wang et al. 2023a) further improved the efficiency by redesigning the masking decoder. It finally became a ViT model with a billion parameters, and it achieved a better result than the original version. Due to the high efficiency of the encoder-decoder and excellent performance on a variety of downstream tasks, many studies have treated VideoMAE v2 as one of the benchmark models.

--> refers to : ASTRA: An Action Spotting TRAnsformer for Soccer Videos [acm](https://dl.acm.org/doi/10.1145/3606038.3616153)

## Others

[Localize humans in crowded space](https://arxiv.org/pdf/2003.07080v1)

[Densely Connected Convolutional Networks](https://www.researchgate.net/publication/306885833_Densely_Connected_Convolutional_Networks)

[Adapting Short-Term Transformers for Action Detection in Untrimmed Videos](https://arxiv.org/abs/2312.01897)

[Spatiotemporal Self-attention Modeling with
Temporal Patch Shift for Action Recognition](https://arxiv.org/pdf/2207.13259)

[Small models : SqueezeNet](https://arxiv.org/pdf/1602.07360)

[Google Learn2compress](https://research.google/blog/custom-on-device-ml-models-with-learn2compress/)

## Biblatex template website

@Online{citekey,
    author = {Author/Organization},
    title = {Title of Webpage},
    url = {URL},
    urldate = {Date Accessed},
    year = {Year Published/ Last Updated}
}
