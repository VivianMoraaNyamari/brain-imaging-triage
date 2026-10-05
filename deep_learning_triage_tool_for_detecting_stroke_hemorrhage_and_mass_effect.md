# A Deep Learning Triage Tool for Detecting Stroke, Hemorrhage, and Mass Effect on Brain CT and MRI to Support Timely Clinical Prioritization

## Abstract

Acute neurological abnormalities such as stroke, hemorrhage, and mass effect require rapid and accurate diagnosis to support timely clinical decision-making and prioritization of care. Brain CT and MRI remain the primary imaging modalities used in emergency and specialist settings, but manual interpretation can be time-consuming and subject to variability. This study proposes the design, development, and comparative evaluation of machine learning and deep learning models for automated detection and triage of stroke, hemorrhage, and mass effect from brain CT and MRI images.

The research compares conventional machine learning approaches, including Logistic Regression, Support Vector Machine (SVM), Random Forest, and XGBoost, using radiomic and quantitative imaging features, against deep learning models based on transfer learning architectures such as ResNet, DenseNet, and EfficientNet. The study further evaluates the performance of these models using standard classification metrics, including accuracy, sensitivity, specificity, precision, F1-score, and area under the receiver operating characteristic curve (AUC-ROC).

To improve clinical interpretability, the study incorporates explainability methods such as SHAP and Grad-CAM, enabling visualization of the imaging features or regions influencing model predictions. A prototype triage decision-support tool is also developed to communicate model outputs and support prioritization of patients with acute neurological findings.

This research contributes to the growing field of AI-assisted neuroimaging by providing a comparative assessment of feature-based and image-based approaches for triage decision support and by demonstrating a practical prototype for clinical prioritization.

---

## 1. Background and Rationale

Brain imaging is essential in the rapid assessment of neurological emergencies. CT and MRI scans provide critical information for identifying conditions such as ischemic stroke, hemorrhage, and mass effect, all of which may require emergent intervention. Early recognition of these abnormalities is central to reducing morbidity and mortality and supporting urgent clinical prioritization.

Despite their value, imaging interpretation is highly dependent on specialist expertise and available clinical resources. In many clinical environments, delays in image review can affect time-sensitive care pathways. Consequently, there is growing interest in artificial intelligence (AI) systems that can support radiologists and clinicians by automating initial triage and detection of critical findings.

Machine learning and deep learning techniques have demonstrated promising performance in medical image classification tasks. Conventional machine learning methods can leverage handcrafted features extracted from images, whereas deep learning models learn hierarchical representations directly from raw imaging data. A comparative evaluation of these approaches is therefore important to determine which strategy offers the best balance of performance, interpretability, and practical usability in a clinical triage context.

---

## 2. Problem Statement

Delayed detection and triage of acute brain abnormalities may lead to missed opportunities for timely intervention. There is a need for robust, accurate, and interpretable AI-assisted systems that can detect stroke, hemorrhage, and mass effect using brain CT and MRI scans.

Although several studies have explored the use of AI for neuroimaging, limited research has focused on a comparative framework that integrates conventional machine learning and deep learning approaches for multi-class triage across CT and MRI modalities. This study addresses that gap by developing and comparing multiple models for automated classification of three key abnormalities: stroke, hemorrhage, and mass effect.

---

## 3. General Objective

To develop and comparatively evaluate machine learning and deep learning models for automated detection and triage of stroke, hemorrhage, and mass effect using brain CT and MRI images.

---

## 4. Specific Objectives

1. To preprocess and characterize brain CT and MRI images for the detection of stroke, hemorrhage, and mass effect.
2. To develop conventional machine learning models using extracted imaging and radiomic features for classification of stroke, hemorrhage, and mass effect.
3. To develop deep learning models using brain CT and MRI images for automated classification of the three abnormalities.
4. To comparatively evaluate the performance of machine learning and deep learning models using appropriate classification metrics.
5. To investigate the explainability of the developed models using techniques such as SHAP and Grad-CAM.
6. To develop a prototype triage decision-support tool that communicates model predictions and prioritization information.

---

## 5. Research Questions

1. How effectively can conventional machine learning models classify stroke, hemorrhage, and mass effect from CT and MRI data?
2. How do deep learning models perform compared with machine learning baselines when using raw brain CT and MRI images?
3. Does the choice of modality (CT vs MRI) affect model performance for detecting the target abnormalities?
4. Can explainability techniques improve transparency and clinical trust in model predictions?
5. Can a prototype triage tool support rapid prioritization of high-risk cases?

---

## 6. Research Hypotheses

H1: Conventional machine learning models trained on radiomic and quantitative imaging features will achieve competitive performance for classification of stroke, hemorrhage, and mass effect.

H2: Deep learning models trained on CT and MRI images will outperform conventional machine learning baselines, especially when transfer learning is used with limited datasets.

H3: CT and MRI modalities will exhibit different strengths depending on the abnormality being detected, with CT likely performing well for hemorrhage and mass effect and MRI offering additional value for ischemic stroke characterization.

H4: Explainability methods such as SHAP and Grad-CAM will provide meaningful insights into model decision-making and improve clinicians' understanding of the outputs.

---

## 7. Scope of the Study

The study focuses on the development and evaluation of AI models for detecting three key brain abnormalities: stroke, hemorrhage, and mass effect. It includes both CT and MRI image datasets and compares the performance of conventional machine learning and deep learning approaches. The study also includes an explainability component and a prototype triage support system.

---

## 8. Significance of the Study

This study is significant for several reasons. First, it addresses a critical clinical need for rapid, automated triage of neurologically urgent findings. Second, it provides a direct comparison between feature-based machine learning models and image-based deep learning models in the same problem domain. Third, it incorporates explainability to support clinical trust and adoption. Finally, it develops a prototype decision-support tool that could serve as a foundation for future clinical deployment or extension to broader emergency neurology workflows.

---

## 9. Methodology

### 9.1 Data Sources

The study will use brain CT and MRI datasets containing cases of stroke, hemorrhage, and mass effect. Datasets may be obtained from publicly available imaging repositories, institutional repositories, or ethically approved hospital collections where accessible.

### 9.2 Data Preprocessing

All images will undergo preprocessing to improve model robustness and consistency. Typical steps may include:

- DICOM import and conversion to a standard format
- Intensity normalization
- Resampling to a uniform voxel size or image dimension
- Skull stripping or region-of-interest extraction where applicable
- Augmentation for deep learning training if needed
- Data splitting into training, validation, and test sets

### 9.3 Feature Extraction for Conventional ML

For conventional machine learning models, quantitative and radiomic features will be extracted from CT and MRI images. These may include:

- Shape features
- Texture features
- Intensity statistics
- Histogram-based features
- Geometric descriptors
- Region-based lesion characteristics

### 9.4 Machine Learning Models

The baseline conventional models will include:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

These models will be trained on extracted imaging features and evaluated on the same classification tasks.

### 9.5 Deep Learning Models

Transfer learning models will be used to process CT and MRI images directly. Candidate architectures include:

- ResNet
- DenseNet
- EfficientNet

These models will be fine-tuned using the study dataset and adapted for the multiclass classification task of identifying stroke, hemorrhage, and mass effect.

### 9.6 Hybrid Model (Optional)

If dataset size and dissertation timelines allow, a hybrid approach may be implemented in which radiomic/clinical features and deep learning features are combined through a fusion classifier. This optional model could provide improved predictive performance over single-modality approaches.

---

## 10. Experimental Design

The study will evaluate the following experiments:

- Experiment A: CT + Machine Learning
- Experiment B: CT + Deep Learning
- Experiment C: MRI + Machine Learning
- Experiment D: MRI + Deep Learning
- Experiment E: CT vs MRI comparison
- Experiment F: Optional multimodal or hybrid model

This structure provides a strong comparative framework without making the dissertation overly complex while still allowing robust evaluation of the research problem.

---

## 11. Model Evaluation Metrics

Model performance will be assessed using standard classification metrics, including:

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-score
- ROC-AUC
- Confusion matrix analysis

For a triage-oriented problem, sensitivity and false-negative rates are especially important, given the clinical implications of missing acute abnormalities.

---

## 12. Explainability and Clinical Interpretability

To support adoption in clinical decision support, model explainability will be evaluated using techniques such as:

- SHAP for feature-level attribution
- Grad-CAM for heatmap visualization of important image regions

These methods will help identify which image regions or features contribute most strongly to the predicted class, thereby improving transparency and interpretability.

---

## 13. Prototype Triage Decision-Support Tool

A prototype system will be developed to present model predictions in a clinically meaningful format. The tool will likely include:

- Patient image upload
- Automated model prediction
- Class output: stroke, hemorrhage, or mass effect
- Confidence score
- Triage prioritization status
- Explainability overlays or supported feature highlights

This prototype is intended to demonstrate how AI output can be communicated to clinicians in a manner that supports urgent prioritization and patient management.

---

## 14. Expected Contributions

This study is expected to contribute to the field by:

1. Providing a comparative evaluation of conventional machine learning and deep learning methods for neuroimaging triage.
2. Demonstrating the value of CT and MRI imaging analysis in detecting acute neurological abnormalities.
3. Highlighting the trade-offs between handcrafted feature-based models and transfer-learning approaches.
4. Integrating explainability into the pipeline to improve clinical interpretability.
5. Delivering a practical prototype that illustrates triage decision support for stroke, hemorrhage, and mass effect.

---

## 15. Dissertation Framing

For an MSc dissertation, the study can be framed as:

"A comparative evaluation of conventional machine learning and deep learning approaches for automated detection and triage of acute neurological abnormalities from brain CT and MRI, accompanied by explainability analysis and prototype implementation."

---

## 16. Conclusion

The proposed study addresses an important problem in neuroimaging and emergency medicine by combining medical image analysis, machine learning, deep learning, explainability, and prototype development. By comparing multiple models on both CT and MRI data, the research aims to provide evidence-based insights into the strengths and limitations of different AI approaches for triage-oriented clinical decision support.

---

## 17. Suggested Dissertation Title Options

1. A Deep Learning Triage Tool for Detecting Stroke, Hemorrhage, and Mass Effect on Brain CT and MRI to Support Timely Clinical Prioritization
2. Comparative Evaluation of Machine Learning and Deep Learning Models for Automated Triage of Stroke, Hemorrhage, and Mass Effect on Brain CT and MRI
3. Automated Detection and Triage of Acute Neurological Abnormalities from Brain CT and MRI Using Machine Learning and Deep Learning
4. Explainable AI for Brain CT and MRI Triage: A Comparative Study of Conventional and Deep Learning Approaches

---

## 18. Short Version for Proposal or Abstract Submission

This study develops and compares machine learning and deep learning models for automated detection and triage of stroke, hemorrhage, and mass effect using brain CT and MRI images. Conventional models such as logistic regression, SVM, random forest, and XGBoost are trained on radiomic and quantitative imaging features, while deep learning models including ResNet, DenseNet, and EfficientNet are trained directly on image data using transfer learning. Model performance is evaluated using standard classification metrics, and explainability methods such as SHAP and Grad-CAM are applied to improve interpretability. A prototype triage decision-support tool is also developed to communicate model predictions and support clinical prioritization. The study aims to provide a practical and comparative assessment of AI approaches for acute neuroimaging triage.

---

## 19. Literature Review Summary

The literature on medical imaging and AI-based triage indicates that machine learning and deep learning methods have become increasingly important for decision support in neuroimaging. Conventional models rely on handcrafted radiomic and statistical feature extraction, which can be advantageous when datasets are small or when clinical interpretability is a priority. Studies have shown that features related to lesion intensity, texture, morphology, and heterogeneity can discriminate between pathological and normal brain tissue with reasonable performance.

Deep learning approaches, in contrast, automatically learn hierarchical feature representations from raw image data and often achieve superior performance in complex imaging classification tasks, particularly when large labeled datasets are available. Transfer learning has emerged as a practical method for medical imaging because it allows the use of pretrained networks on large natural image datasets and adapts them for medical tasks with smaller datasets. This is especially relevant for MSc-level studies where data availability may be limited yet the goal remains to develop clinically useful models.

In the context of brain CT and MRI, AI methods have been used for stroke detection, hemorrhage classification, and mass effect assessment, but there is still a need for broader comparative work that evaluates both modalities and both methodological paradigms within the same study design. This research fills that gap by comparing conventional machine learning with transfer-learning-based deep learning across CT and MRI inputs and by evaluating the explainability of the resulting predictive models.

---

## 20. Conceptual Framework

The conceptual framework for this study is based on the pipeline of medical image analysis, model development, performance evaluation, and clinical interpretation.

At the data layer, CT and MRI images are acquired and preprocessed to ensure consistency. At the feature layer, radiomic features are extracted for traditional machine learning models while raw image data are fed into deep learning architectures. At the model layer, conventional classifiers and transfer-learning models generate predictions for the three target classes: stroke, hemorrhage, and mass effect. At the evaluation layer, performance is measured using classification metrics and comparative analysis. Finally, at the clinical interpretation layer, explainability methods are used to identify the factors contributing to each prediction and to support clinical decision-making through a prototype triage tool.

This framework emphasizes the translation of imaging data into clinically actionable triage outputs and situates the study within the wider discipline of AI-assisted medical decision support.

---

## 21. Variables of the Study

### Independent Variables

- Imaging modality: CT or MRI
- Imaging features used for model development
- Model type: machine learning or deep learning
- Preprocessing methods
- Transfer-learning architecture selected
- Optional fusion of radiomic and deep features

### Dependent Variables

- Predicted class: stroke, hemorrhage, or mass effect
- Classification performance metrics
- Confidence score
- Triage priority classification

### Control Variables

- Dataset splits
- Image size and resolution
- Number of epochs and training parameters
- Augmentation strategies
- Hyperparameter tuning settings

---

## 22. Data Analysis Plan

The study will use a structured data analysis workflow. First, the dataset will be cleaned and organized according to modality and label. Second, preprocessing will be applied consistently across CT and MRI images. Following this, conventional feature extraction and deep-learning model preparation will proceed in parallel.

For machine learning, models will be trained on extracted features and tuned using cross-validation or validation splits. For deep learning, transfer-learning networks will be fine-tuned with appropriate learning rate adjustments, batch sizes, and early stopping criteria to prevent overfitting. Performance will then be compared across the different experiments using the test set.

Statistical comparison between models may include paired comparison of metrics, confidence intervals, and significance testing where appropriate. Model outputs will be analyzed using confusion matrices, ROC curves, and class-wise precision-recall metrics.

---

## 23. Ethical Considerations

This study will adhere to ethical standards for research involving medical data. If using institutional or hospital data, approval must be obtained from the relevant ethics committee or review board. All patient identifiers will be removed or anonymized before analysis. The study will respect patient confidentiality and ensure that data are used strictly for research purposes.

If public datasets are used, only datasets released with appropriate ethical clearance and data-sharing permissions will be considered. The use of AI in healthcare must remain transparent, accountable, and clinically meaningful.

---

## 24. Proposed Timeline and Work Plan

The proposed study will be implemented in stages:

### Stage 1: Problem Definition and Proposal Development
- Finalize research topic and objectives
- Review relevant literature
- Define study design and experimental workflow

### Stage 2: Dataset Acquisition and Preprocessing
- Obtain CT and MRI datasets
- Clean and standardize imaging data
- Split data into training, validation, and test sets

### Stage 3: Feature Extraction and Classical ML Development
- Extract radiomic and quantitative features
- Train Logistic Regression, SVM, Random Forest, and XGBoost
- Evaluate model performance

### Stage 4: Deep Learning Model Development
- Select pretrained architectures such as ResNet, DenseNet, and EfficientNet
- Fine-tune models for the multiclass task
- Evaluate on CT and MRI data

### Stage 5: Explainability and Triage Tool Design
- Apply SHAP and Grad-CAM to interpret predictions
- Design a prototype triage dashboard or tool
- Validate outputs against clinician-oriented requirements

### Stage 6: Analysis, Writing, and Reporting
- Compare model outcomes
- Draft results and discussion chapters
- Complete final dissertation and presentation

---

## 25. Expected Outcomes

This study is expected to produce several outcomes:

1. A comparative evaluation of machine learning and deep learning models for triage classification of stroke, hemorrhage, and mass effect.
2. Evidence on the relative effectiveness of CT and MRI modalities for different pathologies.
3. Insights into the trade-offs between interpretability and predictive performance.
4. An explainability analysis that strengthens clinical understanding of model decisions.
5. A prototype tool that demonstrates how AI could support emergency triage and prioritization.

---

## 26. Limitations of the Study

While the study is expected to provide valuable insights, some limitations are likely. The availability and quality of labeled medical datasets may constrain model performance, particularly for rare neuroimaging conditions. Deep learning models may require substantial computational resources, and class imbalance may affect sensitivity for minority classes. Additionally, an AI prototype may not fully capture all operational and regulatory requirements needed for real-world clinical deployment.

These limitations should be acknowledged and discussed transparently in the final dissertation.

---

## 27. Potential Clinical Impact

The developed system has potential clinical relevance by supporting early detection and triage of severe neurological abnormalities. In emergency settings, even short delays in recognizing life-threatening intracranial pathology can have serious consequences. An AI-supported triage tool could help prioritize urgent cases, streamline clinical workflows, and serve as a second opinion in environments with limited specialist availability.

However, the system should be positioned as a decision-support tool rather than a replacement for clinical judgment. Final diagnosis and treatment decisions should remain the responsibility of trained medical professionals.

---

## 28. Dissertation Chapter Outline

A typical dissertation chapter structure for this study may look as follows:

### Chapter 1: Introduction
- Background
- Problem statement
- Objectives
- Research questions
- Significance of the study

### Chapter 2: Literature Review
- Neurological imaging and clinical relevance
- Stroke, hemorrhage, and mass effect
- Machine learning in medical imaging
- Deep learning and transfer learning in neuroimaging
- Explainability in AI for healthcare

### Chapter 3: Research Methodology
- Data collection and preprocessing
- Feature extraction
- Model development
- Experimental design
- Evaluation metrics

### Chapter 4: Results and Findings
- Model performance across experiments
- Comparative analysis of CT and MRI models
- Explainability results
- Prototype tool evaluation

### Chapter 5: Discussion
- Interpretation of findings
- Comparison with existing literature
- Strengths, limitations, and implications

### Chapter 6: Conclusion and Recommendations
- Summary of findings
- Contributions of the study
- Recommendations for future work

---

## 29. Final Research Positioning

This study occupies a valuable place at the intersection of medical imaging, emergency care, and artificial intelligence. It contributes to the growing body of research that seeks to develop clinically relevant and explainable AI systems for neuroimaging triage. By comparing feature-based and image-based approaches across CT and MRI, the study provides an evidence-based foundation for understanding how AI can support timely detection and prioritization of acute brain abnormalities.

The resulting models and prototype tool are intended not only to perform accurate classification but also to demonstrate a pathway toward meaningful clinical adoption in high-pressure diagnostic environments.

---

## 30. Short Final Summary

This dissertation proposes the design and comparative evaluation of conventional machine learning and deep learning models for detecting stroke, hemorrhage, and mass effect from brain CT and MRI images. The study uses radiomic and statistical features for classical models and transfer learning architectures such as ResNet, DenseNet, and EfficientNet for image-based prediction. Model performance will be evaluated using key clinical metrics, and explainability methods such as SHAP and Grad-CAM will be used to improve trust and transparency. A prototype triage decision-support tool will be developed to demonstrate how the system could support urgent clinical prioritization and decision-making.

---

# Chapter 1: Introduction

## 1.1 Background to the Study

Neurological emergencies represent some of the most time-sensitive and clinically consequential cases encountered in hospital settings. Conditions such as stroke, intracranial hemorrhage, and mass effect can rapidly worsen a patient’s condition if not recognized and managed promptly. Brain CT and MRI are among the most important diagnostic tools used to detect these abnormalities, yet the interpretation of imaging findings requires specialized expertise and often occurs under high-pressure clinical conditions. In emergency departments and tertiary referral centers, clinicians and radiologists must prioritize patients according to the urgency of the diagnosis and potential risk of deterioration.

In recent years, artificial intelligence (AI) has emerged as a promising approach to assist clinicians in medical image interpretation. In neuroimaging, AI systems have been developed to identify tumors, hemorrhage, ischemic lesions, and other abnormalities in CT and MRI images. These systems can support faster triage, improve reproducibility, and reduce the risk of missed diagnoses in settings where radiology workload is high. However, the majority of existing work remains focused on single-pathology detection or single-modality analysis. There is still a need for comparative studies that evaluate both conventional machine learning and deep learning methods across CT and MRI inputs for multi-class triage of acute brain abnormalities.

This study addresses this gap by developing, comparing, and evaluating ML and DL models for the automated detection of stroke, hemorrhage, and mass effect. The research proposes a framework that combines image preprocessing, feature-based classification, deep transfer learning, explainability, and a clinical triage prototype.

## 1.2 Statement of the Problem

The timely detection of intracranial abnormalities is critical to clinical decision-making, but radiological interpretation is often limited by workload, specialist availability, and diagnostic variability. In emergency care, delayed or inaccurate recognition of stroke, intracranial hemorrhage, or mass effect can lead to worse patient outcomes and delayed intervention.

Although CT and MRI are effective tools for identifying such abnormalities, visual interpretation alone may not always permit rapid prioritization of all urgent cases. There is therefore a clear need for intelligent systems capable of supporting early triage and assisting clinicians in identifying patients requiring immediate attention. Existing studies suggest that both conventional machine learning and deep learning can be useful for abnormality detection, yet there is limited evidence comparing these approaches in the same neuroimaging task framework.

This study is designed to address the problem by comparing conventional feature-based classifiers against transfer-learning deep models for the triage of stroke, hemorrhage, and mass effect using CT and MRI data.

## 1.3 Aim of the Study

The aim of this study is to develop and comparatively evaluate machine learning and deep learning models for the automated detection and triage of stroke, hemorrhage, and mass effect from brain CT and MRI images.

## 1.4 Objectives of the Study

The specific objectives of the study are to:

1. Preprocess and characterize brain CT and MRI images for the detection of stroke, hemorrhage, and mass effect.
2. Develop conventional machine learning models using extracted imaging and radiomic features to classify the three abnormalities.
3. Develop deep learning models using CT and MRI images for automated classification of stroke, hemorrhage, and mass effect.
4. Compare the performance of machine learning and deep learning models using appropriate performance metrics.
5. Investigate model explainability using techniques such as SHAP and Grad-CAM.
6. Develop a prototype triage decision-support tool that communicates model predictions and prioritization information.

## 1.5 Research Questions

This study seeks to answer the following questions:

1. How accurately can conventional machine learning models classify stroke, hemorrhage, and mass effect from CT and MRI data?
2. How do deep learning models compare with machine learning baselines for this classification task?
3. Does the imaging modality significantly affect classification performance?
4. Can explainability techniques improve the interpretability and trustworthiness of the model outputs?
5. Can a prototype triage tool provide clinically useful prioritization of high-risk cases?

## 1.6 Hypotheses of the Study

H1: Conventional machine learning models trained on radiomic and imaging features will produce competitive classification performance for stroke, hemorrhage, and mass effect.

H2: Deep learning models using transfer learning will outperform conventional machine learning baselines on CT and MRI image classification.

H3: CT and MRI will show different strengths for different pathology classes, with CT expected to perform strongly for hemorrhage and mass effect and MRI potentially assisting in ischemic stroke characterization.

H4: Explainability methods such as SHAP and Grad-CAM will provide clinically meaningful insight into the decision-making process of the models.

## 1.7 Significance of the Study

This study contributes to the emerging field of AI-assisted neuroimaging by providing a comparison between handcrafted feature-based ML models and deep transfer-learning methods in a clinically relevant and urgent diagnostic context. The research is significant because it supports the development of efficient, explainable, and clinically interpretable triage tools that could assist clinicians in prioritizing high-risk cases.

The study is also relevant to emergency medicine, radiology, and health informatics. In resource-limited settings or high-workload clinical environments, AI-based triage systems can help reduce delays, improve early detection, and support clinical decision-making. The inclusion of explainability methods further increases the practical and ethical value of the work by showing how predictions are derived and where the model focuses within an image.

## 1.8 Scope of the Study

This study is limited to the analysis of brain CT and MRI images for three categories: stroke, hemorrhage, and mass effect. It includes model development using both conventional machine learning and deep learning workflows, comparative evaluation, and prototype implementation. The work focuses on detection and triage support rather than replacing the role of a medical expert.

## 1.9 Definition of Key Terms

- Stroke: A neurological condition caused by impaired blood flow to the brain, which may lead to ischemic or hemorrhagic injury.
- Hemorrhage: Bleeding within or around the brain that may appear as acute blood accumulation on CT or MRI.
- Mass effect: The displacement or distortion of intracranial structures caused by a space-occupying lesion or edema.
- Machine learning: A computational approach that learns patterns from data using engineered or selected features.
- Deep learning: A subset of machine learning that automatically learns hierarchical feature representations from raw image data.
- Transfer learning: The reuse of a model pretrained on large datasets and adapted to a new task with smaller data.
- Explainability: The use of techniques that make model predictions understandable to humans.
- Triage: The prioritization of patients according to urgency and clinical risk.

---

# Chapter 2: Literature Review

## 2.1 Introduction

The literature on medical image analysis has grown substantially in recent years, particularly in the domains of radiology, neurology, and emergency medicine. Advances in computational power, data availability, and algorithmic methods have enabled the development of AI systems capable of identifying clinically significant abnormalities in brain imaging. This literature review examines the role of CT and MRI in detecting stroke, hemorrhage, and mass effect, and it discusses the application of machine learning and deep learning approaches in neuroimaging, especially in relation to triage support.

## 2.2 Stroke, Hemorrhage, and Mass Effect in Brain Imaging

### 2.2.1 Stroke

Stroke is one of the leading causes of morbidity and mortality worldwide. It is often classified into ischemic and hemorrhagic stroke, with each requiring different clinical management. CT is commonly used in the acute setting because it is fast, widely available, and highly effective for identifying hemorrhagic lesions. MRI may provide additional detail for ischemic stroke and can help assess lesion extent, diffusion changes, and tissue viability.

The rapid interpretation of stroke imaging is essential because treatment windows are narrow. Delays in recognition may lead to missed opportunities for thrombolysis or thrombectomy. Accurate and timely triage of suspected stroke cases can therefore materially improve patient outcomes.

### 2.2.2 Hemorrhage

Intracranial hemorrhage is a medical emergency that usually presents as acute bleeding in the brain or surrounding spaces. CT is particularly effective in detecting hyperdense blood collections, while MRI can be useful in characterizing the age and extent of hemorrhage. Early identification of hemorrhage can shorten the time to intervention and reduce the risk of neurological deterioration.

### 2.2.3 Mass Effect

Mass effect refers to the displacement or distortion of brain structures caused by an expanding lesion, edema, hemorrhage, or localized pressure. It is often associated with increased intracranial pressure and can have severe clinical consequences. CT and MRI both contribute to assessing mass effect by depicting shifts, compression, ventricular distortion, and edema patterns. In triage contexts, recognition of mass effect is important because it often implies significant danger and the need for urgent intervention.

## 2.3 Medical Imaging and Clinical Triage

Clinical triage depends on identifying patients who require the highest priority for intervention. Brain CT and MRI support this process by revealing abnormalities that may require emergency management. However, the volume of imaging data, variability in radiologist interpretation, and delays in reporting can hinder triage efficiency. As a result, AI has emerged as a supporting tool for detect-and-prioritize systems that flag possible urgent conditions before definitive radiological review.

## 2.4 Conventional Machine Learning in Neuroimaging

Machine learning techniques have long been used in medical image analysis, especially when features are engineered from image intensities, textures, and shapes. Radiomic features describe properties such as lesion size, intensity distribution, heterogeneity, and spatial variation. These features can be used in algorithms such as Logistic Regression, SVM, Random Forest, and XGBoost.

Conventional models are often attractive because they can be interpreted more easily than deep networks, especially when feature importance is analyzed. They may also perform well on smaller datasets and can be useful when the researcher wishes to understand which imaging characteristics relate to the target class. In stroke and hemorrhage classification, radiomic features such as texture complexity and lesion volume have been used to differentiate pathological tissue from normal tissue.

## 2.5 Deep Learning in Neuroimaging

Deep learning has transformed medical image analysis by learning features directly from raw pixel information. Convolutional Neural Networks (CNNs) have been widely used for image classification, segmentation, and detection tasks. Transfer learning has become a particularly important technique because it allows the use of models pretrained on large benchmark datasets and fine-tunes them for specialized medical tasks.

Popular architectures such as ResNet, DenseNet, and EfficientNet have demonstrated strong performance in imaging tasks because they leverage residual connections, dense connectivity, and efficient depthwise separable convolutions. In brain CT and MRI classification, these models can capture complex visual patterns associated with acute pathology. Transfer learning is particularly useful when medical datasets are limited, a common situation in MSc and clinical research settings.

## 2.6 Comparative Studies in Neuroimaging AI

Several studies have evaluated either machine learning or deep learning models for brain imaging tasks, but fewer studies have directly compared the two paradigms on the same clinical problem. Feature-based methods may achieve competitive accuracy in small datasets, while deep learning often performs better on larger datasets because it captures broader nonlinear relationships. However, deep learning models can be less transparent and more computationally intensive.

This study fills this research gap by comparing both approaches in the specific task of stroke, hemorrhage, and mass effect classification across CT and MRI imaging. The comparative framework allows the researcher to assess not only predictive performance but also practical usefulness in triage scenarios.

## 2.7 Explainability in AI for Healthcare

A major challenge in deploying AI in clinical environments is the need for transparent and explainable decision-making. In healthcare, clinicians must understand why a model makes a prediction, particularly when that prediction affects urgent treatment decisions. Explainability methods address this challenge by highlighting the portions of the input that contributed most to the decision.

Common explainability approaches include:

- SHAP: A model-agnostic method that attributes feature importance to input variables.
- Grad-CAM: A method that visualizes the image regions most influential to a deep-learning prediction.

These techniques are particularly important for medical imaging because they can help clinicians confirm whether the model is focusing on clinically relevant lesions rather than irrelevant artifacts. In the present study, explainability is used to increase confidence in AI-assisted triage and to support human oversight in clinical interpretation.

## 2.8 Summary of the Literature Review

The literature demonstrates that CT and MRI are central to the diagnosis of acute neurological abnormalities, and that AI-based systems hold promise for enhancing triage and decision support. Conventional machine learning remains valuable, especially when features are well engineered and datasets are limited. Deep learning, particularly transfer learning, offers strong performance for raw image classification but requires more data and computational resources.

A comparative evaluation of both approaches across CT and MRI for stroke, hemorrhage, and mass effect is therefore highly relevant. Such a study is useful for identifying the best-performing strategy and for understanding how explainability can support safe and clinically meaningful deployment.

---

# Chapter 3: Research Methodology

## 3.1 Research Design

This study adopts a quantitative, comparative, experimental research design. It focuses on the development and evaluation of machine learning and deep learning models for multiclass classification of brain CT and MRI images into three target classes: stroke, hemorrhage, and mass effect. The study includes both conventional feature-based models and transfer-learning image-based models, allowing direct comparison of predictive performance and interpretability.

## 3.2 Study Population and Data Sources

The study population consists of brain CT and MRI images obtained from patients diagnosed with one of the three target conditions. The dataset may be sourced from public repositories, institutional databases, or ethically approved clinical collections, provided that the data are appropriately anonymized and accessible for research use.

The key requirement is that each image is labeled with the correct pathologic class, and that both CT and MRI modalities are available or separately analyzed according to experimental design.

## 3.3 Data Collection Procedure

Data collection will involve identifying suitable CT and MRI datasets that include annotated cases of stroke, hemorrhage, and mass effect. The images will be organized into structured folders according to modality and class label. Metadata associated with each image, such as patient ID, study description, and imaging protocol, will also be reviewed where available.

To ensure data quality, images with severe artifacts, incomplete metadata, or inconsistent labeling will be excluded or reviewed by a clinical expert if available.

## 3.4 Data Preprocessing

Before model development, all images will be preprocessed to standardize the input format and improve model training stability. Typical preprocessing steps may include:

- Conversion of files to a consistent image format
- Resampling to a uniform spatial resolution
- Intensity normalization or windowing
- Cropping or region-of-interest extraction around the brain region
- Removal of irrelevant background or artifacts
- Image augmentation for deep learning training, where appropriate
- Partitioning into training, validation, and test subsets

For deep learning, images may be resized to match the required input size of the selected pretrained network, such as 224x224 or 299x299 pixels depending on the architecture.

## 3.5 Feature Extraction for Conventional Machine Learning

Conventional machine learning models require engineered features rather than raw pixel data. Relevant features will be extracted from brain CT and MRI images, including:

- Intensity statistics (mean, variance, skewness, kurtosis)
- Texture features (contrast, entropy, homogeneity, correlation)
- Shape and size features
- Histogram-based features
- Morphological descriptors of lesions
- Radiomic features derived from regions of interest

These features will be compiled into tabular datasets for model training. Feature selection or dimensionality reduction may be applied where necessary to reduce redundancy and improve model performance.

## 3.6 Machine Learning Models

The conventional models to be used in this study are:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

These models will be trained on the extracted features to classify the images into stroke, hemorrhage, or mass effect. Hyperparameter optimization will be performed using validation data to improve generalization and reduce overfitting.

## 3.7 Deep Learning Models

Deep learning models will use raw CT and MRI images as input. Transfer learning will be used to leverage pretrained architectures, which is appropriate for small to medium-sized medical datasets. Candidate architectures include:

- ResNet
- DenseNet
- EfficientNet

The selected pretrained models will be adapted to a three-class classification problem by replacing the final classification layer and fine-tuning the network using the study dataset. Optimization strategies may include adaptive learning rate schedules, early stopping, and regularization to prevent overfitting.

## 3.8 Hybrid Model (Optional)

If the dataset size and project timeline permit, a hybrid model may be developed by combining radiomic features and deep-learning features into a fusion classifier. This approach may help capture both handcrafted medical-domain features and high-level representation learning. The hybrid model could serve as an exploratory extension beyond the main comparison.

## 3.9 Experimental Setup

The study will be organized into the following experiments:

- Experiment A: CT + machine learning
- Experiment B: CT + deep learning
- Experiment C: MRI + machine learning
- Experiment D: MRI + deep learning
- Experiment E: CT vs MRI comparison
- Experiment F: Hybrid or multimodal model (optional)

This design enables a systematic comparison of modality-level and model-type differences while keeping the overall thesis manageable and focused.

## 3.10 Model Evaluation

The performance of the models will be evaluated using the following metrics:

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-score
- Area Under the Curve (AUC)
- Confusion matrix analysis
- ROC curves

Because this is a clinical triage task, sensitivity and false-negative rate are especially important. A high false-negative rate would be clinically unacceptable for acute neuroimaging findings.

## 3.11 Explainability Methods

To improve transparency and trustworthiness, explainability techniques will be applied after model training. These will include:

- SHAP for feature-level interpretation in conventional models
- Grad-CAM for visual localization in deep learning models

The explainability outputs will be examined to determine whether the model is focusing on clinically relevant areas of the brain or on irrelevant patterns. This is particularly important in medical AI, where interpretability is a key requirement for eventual adoption.

## 3.12 Prototype Triage Tool Development

A prototype triage decision-support tool will be designed based on the best-performing model. The tool could include the following components:

- Image upload function
- Model prediction output
- Probabilistic confidence score
- Class label: stroke, hemorrhage, or mass effect
- Triage recommendation such as urgent, high priority, or standard review
- Explainability visualization such as heatmap or feature importance chart

The prototype will be designed to demonstrate how model outputs can be translated into a clinically relevant decision-support system without replacing clinician judgment.

## 3.13 Data Analysis Procedures

The collected data will be analyzed using a standard workflow:

1. Data preparation and cleaning
2. Preprocessing and augmentation
3. Feature extraction for ML models
4. Training of ML and DL models
5. Validation and hyperparameter optimization
6. Evaluation on held-out test data
7. Statistical comparison of model performance
8. Explainability analysis
9. Prototype tool development and discussion

All results will be summarized in tables and figures to facilitate interpretation and comparison across experiments.

## 3.14 Validity and Reliability

To improve the reliability of the study, data splitting will be performed carefully to avoid information leakage between training and testing sets. Cross-validation or validation sets will be used for model tuning. The same evaluation criteria will be applied across all model types to ensure a fair comparison. Additionally, repeated experiments or multiple random seeds may be used where feasible to assess model stability.

## 3.15 Ethical Considerations

This research will comply with ethical standards for medical data use. If identifiable information is present, it will be anonymized or removed before processing. Approval will be sought from the relevant institutional ethics committee where required. The study will ensure that data are used only for research and that patient privacy is protected at all stages.

## 3.16 Summary of Methodology

This chapter has outlined the overall methodological framework for the study. The research adopts a comparative and experimental design that includes both conventional machine learning and deep learning approaches. CT and MRI images will be preprocessed, labeled, and analyzed using a multistage workflow that includes feature engineering, transfer learning, performance evaluation, and explainability analysis. The resulting model outcomes will be used to develop a triage-oriented prototype that supports early detection and clinical prioritization.

---

## 31. Final Thesis-Ready Summary

This study addresses an important clinical challenge in emergency neuroimaging by exploring how machine learning and deep learning can support the detection and triage of acute brain abnormalities. Stroke, hemorrhage, and mass effect are time-critical conditions that require rapid recognition and prioritization. Brain CT and MRI provide the primary imaging basis for diagnosis, but manual interpretation can be constrained by time pressure, limited specialist capacity, and variability in reporting.

The proposed research compares feature-driven conventional models with transfer-learning-based deep models across CT and MRI datasets. It examines performance through standard evaluation metrics, explores model explainability, and develops a prototype triage support tool. The overall significance of the study lies in its practical relevance, comparative design, and alignment with real clinical needs for early and reliable prioritization of neurologically urgent patients.

This dissertation will contribute to the growing field of artificial intelligence in medical imaging by providing evidence on the relative strengths of conventional and deep learning approaches for emergency brain triage, while emphasizing interpretability, safety, and clinical usefulness.
# Chapter 4: Results and Findings

## 4.1 Introduction

This chapter presents the expected results and findings of the study based on the comparative evaluation of machine learning and deep learning models for stroke, hemorrhage, and mass effect classification in brain CT and MRI images. The chapter outlines how the performance of the developed models will be reported, compared, and interpreted in relation to the objectives of the study.

## 4.2 Data Distribution and Experimental Setup

The dataset will be divided into training, validation, and test subsets, ensuring that the model evaluation is unbiased and clinically meaningful. The class distribution will be reviewed to identify any imbalance between stroke, hemorrhage, and mass effect cases. If necessary, balancing strategies such as class weighting, oversampling, or augmentation may be used to improve model learning.

The experimental pipeline will include preprocessing, feature extraction, model training, validation, and final testing. The performance of each model will be assessed on the hold-out test set to avoid information leakage.

## 4.3 Baseline Model Performance

The conventional machine learning models will be evaluated using extracted radiomic and quantitative imaging features. The expected findings may show that models such as Logistic Regression, SVM, Random Forest, and XGBoost perform reasonably well, especially where well-defined lesion characteristics are available. SVM and XGBoost may perform strongly because they are capable of modeling nonlinear boundaries and handling diverse feature sets.

Key metrics such as accuracy, F1-score, sensitivity, and ROC-AUC will be reported for each model. The strongest classical model is expected to provide a reliable baseline for comparison with deep learning.

## 4.4 Deep Learning Performance

The deep learning models trained on raw CT and MRI images are expected to outperform baseline classical models when sufficient training data are available. Transfer learning architectures such as ResNet, DenseNet, and EfficientNet are likely to capture complex spatial and intensity patterns relevant to acute neurological abnormalities.

The results may show that EfficientNet performs competitively because of its efficient scaling and high representational power, while DenseNet may be especially strong because of feature reuse. ResNet may also produce robust performance because of the residual architecture that supports efficient training on deeper networks.

## 4.5 CT Versus MRI Comparison

A major part of the study will focus on comparing model performance across imaging modalities. It is expected that CT may demonstrate exceptional performance for hemorrhage detection because acute blood appears hyperdense on CT and can be readily distinguished. MRI may provide better sensitivity for ischemic stroke and subtle tissue changes, although it may also be more variable due to acquisition differences.

The comparison between CT and MRI will therefore not identify a universally superior modality, but rather highlight modality-specific strengths depending on the target condition. This is clinically relevant because imaging modality selection influences both diagnosis and triage prioritization.

## 4.6 Explainability Results

The explainability analysis will be used to identify which image regions or features are contributing most to each prediction. For conventional models, SHAP values may reveal the roles of lesion size, texture, and intensity in determining the class. For deep learning models, Grad-CAM heatmaps will show the parts of the image that the network focused on during classification.

The expected finding is that clinically relevant regions such as hemorrhagic regions, infarct zones, or areas of mass effect will receive higher attribution scores than irrelevant background regions. If the model focuses on clinically meaningful structures, the resulting explanation will support confidence in the system and improve trust among clinicians.

## 4.7 Prototype Triage Tool Outputs

The prototype triage decision-support tool is expected to translate the model outputs into clinically usable categories. For example:

- Stroke: urgent review
- Hemorrhage: immediate prioritization
- Mass effect: critical review
- Confidence indicator: low, moderate, or high

This tool is not intended to replace medical professionals but to support the triage workflow by highlighting high-priority cases. A successful prototype would show that the model can rapidly identify cases requiring immediate attention and communicate the result in a way that is understandable to clinicians.

## 4.8 Summary of Findings

The results of this study are expected to demonstrate that:

1. Machine learning models can provide useful and interpretable baseline performance.
2. Deep learning models can achieve higher predictive accuracy using raw image data and transfer learning.
3. CT and MRI exhibit different strengths depending on the pathology.
4. Explainability techniques can reveal clinically meaningful patterns.
5. A triage decision-support tool can provide a practical clinical interface for AI output.

---

# Chapter 5: Discussion

## 5.1 Interpretation of Findings

The discussion chapter will interpret the results in the context of the study objectives and the broader medical imaging literature. If deep learning models outperform conventional classifiers, this would support the argument that image-based learning captures richer, more complex patterns of acute brain abnormalities than handcrafted feature engineering alone. However, if feature-based models perform competitively, this would suggest that radiomic descriptors remain highly useful, particularly when datasets are small or when explainability is essential.

The findings are likely to emphasize that model performance depends not only on the algorithm but also on the imaging modality, the nature of the pathology, and the quality of preprocessing. In practical terms, the choice of model should reflect the clinical task, available data, and the need for transparency.

## 5.2 Comparison with Existing Literature

The results will be compared with previous studies in neuroimaging, radiomics, and deep learning. The expected contribution of this study is its comparative design: it evaluates both machine learning and deep learning within the same multi-class triage problem and across CT and MRI modalities. This approach is more informative than isolated single-modality studies because it provides a direct understanding of the relative strengths of different modeling strategies.

The study is also likely to align with prior evidence that transfer learning is highly effective in medical imaging when labeled data are limited. At the same time, the study acknowledges the continuing importance of handcrafted features and model explainability, particularly in clinically sensitive settings.

## 5.3 Implications for Clinical Practice

The clinical implications of this study are significant. A system capable of rapidly detecting critical brain abnormalities could support the triage process in emergency departments and specialist imaging environments. Such a system could reduce reporting delays, improve consistency, and improve prioritization of patients in urgent need of intervention.

However, the model should not be viewed as a replacement for medical professionals. Instead, it should serve as a decision-support assistant that highlights suspected abnormalities, accelerates review, and supports clinical reasoning. In this sense, the AI tool complements specialist expertise rather than substituting for it.

## 5.4 Strengths of the Study

The study has several strengths:

- It addresses a clinically important and urgent problem.
- It compares both machine learning and deep learning paradigms.
- It includes both CT and MRI modalities.
- It evaluates explainability, which is crucial for medical applications.
- It includes a prototype decision-support tool relevant to triage workflows.

These strengths make the study both academically valuable and practically relevant.

## 5.5 Limitations of the Study

Despite its strengths, the research will have limitations. Medical datasets may be imbalanced, and pathology-specific cases may be underrepresented. Deep learning models may require substantial computational resources and careful hyperparameter tuning. A prototype system may not fully reflect real-world clinical deployment conditions or hospital workflows. In addition, image biases, acquisition variability, and dataset heterogeneity may affect generalizability.

These limitations should be acknowledged transparently and incorporated into the interpretation of the findings.

## 5.6 Recommendations for Future Work

Future work may include:

- Expansion to larger and more diverse datasets
- Inclusion of additional abnormalities or disease categories
- Development of a multimodal fusion model combining CT, MRI, and clinical variables
- Integration of segmentation models to isolate lesions before classification
- Deployment in real clinical settings with usability evaluation by clinicians
- Formal regulatory and ethical assessment before clinical implementation

These future directions would strengthen the real-world applicability of the system and support its translation from research to practice.

---

# Chapter 6: Conclusion and Recommendations

## 6.1 Conclusion

This dissertation has proposed the development and comparative evaluation of machine learning and deep learning models for automated classification and triage of stroke, hemorrhage, and mass effect from brain CT and MRI images. The study addresses a clinically important problem by positioning AI as a decision-support tool in emergency neuroimaging.

The findings are expected to show that both conventional and deep learning models can contribute meaningfully to the task, but their relative performance depends on modality, data availability, and the balance between predictive performance and interpretability. Deep learning is likely to offer strong predictive power, while conventional models may remain attractive because of their transparency and practical utility in smaller datasets.

By incorporating explainability and a triage prototype, the study also moves beyond simple classification and explores how AI can be embedded into clinically useful workflows. This makes the research particularly relevant for healthcare systems that seek efficient and interpretable support tools.

## 6.2 Recommendations

It is recommended that:

1. Future neuroimaging triage systems adopt a comparative framework that evaluates multiple models rather than relying on a single algorithm.
2. Model interpretability be treated as a core requirement in clinical AI design, not an optional feature.
3. CT and MRI performance be assessed separately and jointly to identify the most appropriate modality for each pathology.
4. Prototype tools be validated by clinicians to assess usability, trust, and workflow integration.
5. Future studies expand datasets and clinical validation to support safer deployment.

## 6.3 Contribution to Knowledge

This study contributes to the body of knowledge by providing a comparative evaluation of conventional and deep learning approaches for brain CT and MRI triage, with a specific focus on stroke, hemorrhage, and mass effect. It highlights the value of explainability and prototype development in the deployment of AI-based medical decision support, which remains an underexplored but critical area in clinical informatics.

The work is expected to be useful to researchers, radiologists, clinicians, and healthcare technology developers interested in AI-supported triage systems for emergency neuroimaging.

---

## 32. Final Research Statement

The completed dissertation will present an integrated framework for developing and evaluating AI-based triage systems for severe brain abnormalities using CT and MRI imaging. In doing so, it addresses the clinical need for rapid, accurate, and interpretable decision support in emergency care. The study’s value lies not only in performance comparison but also in its potential to translate research into a usable prototype that supports timely prioritization and improves patient outcomes.

---

## 33. Summary of Expected Outcomes and Comparison Matrix

| Experiment | Modality | Model Type | Expected Outcome | Primary Metric |
|---|---|---|---|---|
| A | CT | Machine Learning | Strong baseline performance for hemorrhage and mass effect detection | Accuracy, F1-score |
| B | CT | Deep Learning | High sensitivity for acute abnormalities with transfer learning | AUC, recall |
| C | MRI | Machine Learning | Useful performance for ischemic stroke and lesion heterogeneity | Accuracy, precision |
| D | MRI | Deep Learning | Strong performance for subtle lesion patterns and tissue characterization | F1-score, ROC-AUC |
| E | CT vs MRI | Comparative | Highlights modality-specific strengths and weaknesses | Sensitivity, specificity |
| F | Hybrid | Fusion classifier | Potentially improved performance by combining handcrafted and deep features | ROC-AUC, accuracy |

The comparative matrix above reflects the conceptual structure of the study and illustrates how each experimental setting contributes to the overall research objectives. The most appropriate model may not be uniform across all classes, so performance will be examined at both the aggregate and class-specific levels.

---

## 34. One-Page Supervisor Summary

This study investigates the development and evaluation of machine learning and deep learning models for the automated detection and triage of stroke, hemorrhage, and mass effect in brain CT and MRI images. The research is motivated by the urgent need for rapid prioritization of life-threatening neurological abnormalities in clinical settings, where delays in interpretation can affect patient outcomes. The study compares conventional feature-based classifiers, including Logistic Regression, SVM, Random Forest, and XGBoost, with transfer-learning deep models such as ResNet, DenseNet, and EfficientNet.

The project adopts a multi-stage approach involving image preprocessing, feature extraction, model training, evaluation, explainability analysis, and prototype development. It compares CT and MRI workflows separately and in combination to determine which modality and model architecture provide the strongest decision-support performance. Model performance will be assessed using standard classification metrics, including accuracy, sensitivity, specificity, precision, F1-score, and ROC-AUC.

To improve clinical usability, the study incorporates explainability methods such as SHAP and Grad-CAM, making it possible to understand which image regions or features drive the prediction. A prototype triage decision-support tool will also be developed to translate model output into a clinically interpretable prioritization system. This component is important because the goal is not only to classify abnormalities but also to demonstrate how such tools could assist clinicians in emergency settings.

The study is significant because it offers a practical comparison of feature-based and image-based modeling strategies within a clinically relevant neuroimaging context. It contributes to the growing literature on AI-based medical imaging by evaluating not only predictive performance but also transparency, interpretability, and triage value. The expected outcomes include a rigorous comparative assessment of model performance, deeper understanding of CT and MRI modality differences, and a prototype system that supports urgent clinical prioritization.

---

## 35. Final Abstract for Submission

Acute neurological abnormalities such as stroke, hemorrhage, and mass effect require rapid diagnosis and prioritization to reduce morbidity and improve patient outcomes. This study develops and comparatively evaluates conventional machine learning and deep learning models for automated detection and triage of these abnormalities using brain CT and MRI images. Conventional models including Logistic Regression, Support Vector Machine, Random Forest, and XGBoost are trained on radiomic and quantitative imaging features, while deep learning models based on transfer learning architectures such as ResNet, DenseNet, and EfficientNet are trained directly on image data.

The study compares model performance across CT and MRI modalities using standard classification metrics, including accuracy, precision, recall, specificity, F1-score, and ROC-AUC. To improve interpretability and clinical trust, explainability methods such as SHAP and Grad-CAM are used to identify the features and image regions most associated with model predictions. A prototype decision-support triage tool is also developed to communicate model outputs in a clinically meaningful and prioritized manner.

The project is significant because it addresses a time-critical clinical need while providing a clear comparative evaluation of model types, imaging modalities, and explainability strategies. By integrating machine learning, deep learning, neuroimaging, and decision-support design, the study contributes to the growing field of AI-assisted diagnostic systems for emergency brain triage.

---

## 36. Final Proposed Title

A Deep Learning Triage Tool for Detecting Stroke, Hemorrhage, and Mass Effect on Brain CT and MRI to Support Timely Clinical Prioritization

This title remains concise, clinically relevant, and suitable for both a thesis and a presentation context.

---

## 37. Prototype System Architecture and Implementation Plan

The prototype triage tool developed in this study will serve as a practical demonstration of how the trained AI models can be integrated into a clinician-facing decision-support workflow. The design will focus on usability, interpretability, and time efficiency, while ensuring that the system remains aligned with evidence-based clinical practice.

### 37.1 System Overview

The proposed system will comprise the following major components:

1. Data ingestion module
   - Accepts CT or MRI images in common formats such as DICOM or PNG/JPG
   - Validates the file type and image quality
   - Loads patient study metadata where available

2. Preprocessing pipeline
   - Standardizes image dimensions and resolution
   - Applies intensity normalization, grayscale conversion, and optional cropping
   - Prepares the image for model inference

3. Prediction engine
   - Loads the trained classical ML model or deep-learning model
   - Runs inference on the uploaded image
   - Produces class probabilities for stroke, hemorrhage, and mass effect

4. Explainability module
   - Uses SHAP for feature-based models
   - Uses Grad-CAM or similar saliency mapping for deep-learning outputs
   - Highlights the image region influencing the decision

5. Triage decision layer
   - Maps the model output to a priority category such as emergency, urgent review, or routine follow-up
   - Displays confidence level and recommended action

6. User interface
   - Presents results through a simple dashboard or web application
   - Allows clinicians to inspect the prediction and explanation visually

### 37.2 Recommended Technology Stack

A practical implementation may use the following tools:

- Python as the primary programming language
- PyTorch or TensorFlow/Keras for deep learning
- scikit-learn for classical machine learning models
- SimpleITK or OpenCV for image preprocessing and augmentation
- pydicom for DICOM reading and conversion
- SHAP for explainability in classical models
- Grad-CAM libraries or custom implementation for CNN visualization
- Streamlit or Flask for the prototype user interface

This stack is appropriate for a research prototype because it is widely used, accessible, and suitable for rapid iterative development.

### 37.3 Implementation Workflow

The prototype implementation will follow a structured workflow:

1. User uploads a CT or MRI image.
2. The system validates the file and loads it into the preprocessing pipeline.
3. The chosen trained model performs inference.
4. Prediction probability scores are computed for each class.
5. The class with the highest confidence is displayed as the predicted diagnosis.
6. The explainability module overlays relevant regions or highlights important features.
7. The triage layer produces a short prioritization message.
8. The clinician reviews the result alongside the visual explanation.

This workflow reflects the central goal of the study: to support clinical prioritization rather than replace clinical judgment.

### 37.4 Example Triage Output Logic

A simple tiered logic may be defined as follows:

- Hemorrhage with confidence above threshold → Immediate prioritization
- Mass effect with confidence above threshold → Urgent review
- Stroke with confidence above threshold → High-priority attention
- Low-confidence or uncertain cases → Secondary review by specialist

This logic is useful because the clinical importance of a false negative is higher than that of a false positive in a triage context.

### 37.5 Usability Considerations

The prototype should be designed with clinical usability in mind. Recommendations include:

- Simple interface with minimal training required
- Clear labels for prediction and confidence
- Visual explanations that are easy to interpret
- Low latency for real-time or near-real-time inference
- Avoidance of unnecessary technical complexity

These considerations are essential to ensure that the tool is clinically understandable and supports workflow efficiency in urgent settings.

### 37.6 Practical Deployment Considerations

Although the system is developed as a prototype, several deployment issues should be considered:

- Secure storage of medical images and patient data
- Compliance with institutional data protection policies
- Integration with hospital information systems if expanded
- Continuous monitoring of model performance after deployment
- Periodic retraining with new retrospective or prospective data

These issues are important for moving from a research prototype toward an operational clinical tool.

---

## 38. Ethical, Feasibility, and Risk Assessment

The deployment of AI systems in medical imaging raises ethical and operational considerations. This section summarizes key issues relevant to the proposed research.

### 38.1 Ethical Risks

- Risk of bias if the dataset is not representative of the target population
- Risk of over-reliance on automated predictions by clinicians
- Risk of misclassification in acute clinical situations
- Risk of privacy violations if patient data are not properly anonymized

### 38.2 Feasibility Risks

- Dataset imbalance across classes
- Limited availability of labeled medical images
- Computational limitations for deep-learning training
- Variability in image acquisition protocols and scanner types

### 38.3 Mitigation Strategies

- Use balanced class weighting or augmentation where needed
- Validate the model on held-out test data
- Include explainability to improve clinical oversight
- Use anonymized or ethically approved data only
- Interpret model output as decision support and not final clinical diagnosis

This framework ensures that the project remains both scientifically rigorous and clinically responsible.

---

## 39. Final Summary for the Research Proposal

This study proposes a comparative and clinically relevant investigation into AI-driven triage of neurological abnormalities using brain CT and MRI. The research combines classical machine learning, deep transfer learning, explainability, and prototype design within a single framework. Its key contribution is not only to compare predictive performance but also to demonstrate how artificial intelligence can support emergency triage, improve prioritization, and assist clinicians in time-sensitive decision-making.

The work is aligned with the current direction of medical AI research and is positioned to make a meaningful contribution to both academic knowledge and practical clinical decision support.

---

