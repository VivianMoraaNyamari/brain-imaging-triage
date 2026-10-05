# Brain Imaging Triage Workflow

```mermaid
flowchart TD
    A[Brain Imaging Data] --> B[CT]
    A --> C[MRI]

    B --> D[Image Preprocessing]
    C --> D

    D --> E[Feature Extraction]
    D --> F[Raw Images]

    E --> G[Radiomic / Statistical Features]
    G --> H[ML Models]
    H --> I[RF]
    H --> J[XGBoost]
    H --> K[SVM]

    F --> L[Deep Learning]
    L --> M[ResNet]
    L --> N[DenseNet]
    L --> O[EfficientNet]

    I --> P[Comparative Evaluation]
    J --> P
    K --> P
    M --> P
    N --> P
    O --> P

    P --> Q[Performance Evaluation]
    P --> R[Explainability]
    R --> S[SHAP]
    R --> T[Grad-CAM]

    Q --> U[Triaging Prototype]
    S --> U
    T --> U

    U --> V[Stroke Hemorrhage]
    U --> W[Mass Effect]

    V --> X[Clinical Prioritization]
    W --> X

    classDef main fill:#E8F1FF,stroke:#4A6FA5,stroke-width:1px,color:#1F2A44;
    classDef model fill:#EAF9EA,stroke:#3A8D66,stroke-width:1px,color:#1F2A44;
    classDef eval fill:#FFF0D6,stroke:#C18B2D,stroke-width:1px,color:#1F2A44;
    classDef triage fill:#FDE9EC,stroke:#B85C6C,stroke-width:1px,color:#1F2A44;

    class A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X main;
    class I,J,K,M,N,O model;
    class Q,R,S,T eval;
    class U,V,W,X triage;
```

This Mermaid version recreates the architecture shown in the provided diagram for a brain imaging triage system using CT/MRI data, preprocessing, ML and deep learning models, evaluation, explainability, and clinical prioritization.
