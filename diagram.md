graph TD
    A[HAM10000 Dataset<br>10,000 Dermatoscopic Images] --> B[20% Stratified Subsample<br>~2,003 Images]
    B --> C[70% Train (1,402) / 15% Val (300) / 15% Test (301)]
    
    C --> D[Branch A: Classical ML Pipeline]
    C --> E[Branch B: Deep Learning Pipeline]
    
    D --> F[OpenCV Feature Extraction<br>60 Features: 48 HSV Histograms + 12 Stats]
    F --> G[Baseline Random Forest<br>Unweighted]
    F --> H[Balanced Random Forest<br>class_weight='balanced']
    
    E --> I[PyTorch Custom Dataset<br>Resizing & Augmentations]
    I --> J[MobileNetV2 Transfer Learning<br>Pre-trained on ImageNet]
    J --> K[Weighted Cross-Entropy Loss<br>Inverse Class Frequencies]
    
    G --> L[Model Evaluation & Confusion Matrices]
    H --> L
    K --> L
    
    L --> M[Final Results & Comparison<br>MobileNetV2 Macro F1: 0.4849]