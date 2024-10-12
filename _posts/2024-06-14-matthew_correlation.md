---
author_profile: false
categories:
- Machine Learning
classes: wide
date: '2024-06-14'
excerpt: Dive deep into Matthew's Correlation Coefficient (MCC), a powerful metric for evaluating binary classification models, especially in imbalanced datasets.
header:
  image: /assets/images/data_science_2.jpg
  og_image: /assets/images/data_science_7.jpg
  overlay_image: /assets/images/data_science_2.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_2.jpg
  twitter_image: /assets/images/data_science_7.jpg
keywords:
- Mcc
- Matthew’s correlation coefficient
- Binary classification
- Confusion matrix
- Model evaluation
- Imbalanced datasets
- Machine learning metrics
- Python
- Fortran
- Sh
- C
- Mathematics
- Statistics
- Data science
- python
- fortran
- sh
- c
seo_description: Learn about Matthew’s Correlation Coefficient (MCC), an essential metric for evaluating binary classification models, particularly in imbalanced datasets, and how it improves upon traditional metrics.
seo_title: 'Matthew’s Correlation Coefficient (MCC): A Guide to Binary Classification'
seo_type: article
subtitle: Understanding and Applying MCC in Binary Classification
summary: This article provides a comprehensive explanation of Matthew’s Correlation Coefficient (MCC), its importance in binary classification, and how it compares to other performance metrics like accuracy, precision, and recall.
tags:
- Mcc
- Evaluation metrics
- Binary classification
- Machine learning
- Statistical methods
- Confusion matrix
- Predictive modeling
- Performance metrics
- Data analysis
- Python
- Fortran
- Sh
- C
- python
- fortran
- sh
- c
title: 'Matthew’s Correlation Coefficient (MCC): A Detailed Explanation'
---

## Introduction

### Definition of MCC

Matthew’s Correlation Coefficient (MCC) is a statistical measure used to evaluate the quality of binary classifications. Unlike other metrics, MCC takes into account all four categories of a confusion matrix—true positives, true negatives, false positives, and false negatives—providing a balanced measure even when classes are of very different sizes.

### Importance

The significance of MCC lies in its ability to provide a more comprehensive evaluation of binary classification models. While metrics like accuracy can be misleading, especially with imbalanced datasets, MCC offers a single value that encapsulates the performance across all classes. This makes it a robust measure for assessing the effectiveness of a model in distinguishing between the two classes.

### Objective

In this article, we will delve into the details of Matthew’s Correlation Coefficient. We will explore its mathematical formulation, interpret its values, and discuss its advantages and limitations. Additionally, we will compare MCC with other commonly used metrics and provide practical examples to illustrate its application in real-world scenarios. By the end of this article, you will have a thorough understanding of MCC and its relevance in binary classification tasks.

## What is MCC?

### Definition

Matthew’s Correlation Coefficient (MCC) is a statistical measure used to evaluate the performance of binary classification models. It considers all four categories of the confusion matrix—true positives, true negatives, false positives, and false negatives. By incorporating these values, MCC provides a single metric that captures the balance between the positive and negative classes, making it especially useful for imbalanced datasets.

### Historical Background

MCC was introduced by the British biochemist Brian W. Matthews in 1975. Initially used in the field of bioinformatics, it has since become a standard measure in machine learning and statistics for evaluating binary classifiers.

### Significance

The importance of MCC in machine learning and statistics stems from its comprehensive nature. Unlike other metrics, such as accuracy or precision, MCC provides a more balanced evaluation by considering both false positives and false negatives. This makes it a crucial metric for applications where the cost of misclassification is high or when dealing with imbalanced datasets.

## Mathematical Formulation

### Formula

The formula for Matthew’s Correlation Coefficient (MCC) is given by:

$$\text{MCC} = \frac{(TP \times TN) - (FP \times FN)}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

### Components

#### True Positives (TP)

True Positives are instances where the model correctly predicts the positive class. These are cases where the actual class is positive, and the model also classifies it as positive.

#### True Negatives (TN)

True Negatives are instances where the model correctly predicts the negative class. These are cases where the actual class is negative, and the model also classifies it as negative.

#### False Positives (FP)

False Positives are instances where the model incorrectly predicts the positive class. These are cases where the actual class is negative, but the model classifies it as positive.

#### False Negatives (FN)

False Negatives are instances where the model incorrectly predicts the negative class. These are cases where the actual class is positive, but the model classifies it as negative.

## Example Calculation

### Hypothetical Data

Let's consider a binary classification problem with the following confusion matrix:

- True Positives (TP): 50
- True Negatives (TN): 30
- False Positives (FP): 10
- False Negatives (FN): 5

### Calculation

Using the MCC formula:

$$\text{MCC} = \frac{(TP \times TN) - (FP \times FN)}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

Substitute the values:

$$\text{MCC} = \frac{(50 \times 30) - (10 \times 5)}{\sqrt{(50 + 10)(50 + 5)(30 + 10)(30 + 5)}}$$

$$\text{MCC} = \frac{1500 - 50}{\sqrt{60 \times 55 \times 40 \times 35}}$$

$$\text{MCC} = \frac{1450}{\sqrt{4620000}}$$

$$\text{MCC} = \frac{1450}{2149.42}$$

$$\text{MCC} \approx 0.675$$

### Interpretation

The MCC value of approximately 0.675 indicates a strong positive correlation between the predicted and actual classifications. This suggests that the model is performing well, with a balanced accuracy in distinguishing between the positive and negative classes. An MCC value closer to 1 implies better performance, while a value closer to -1 indicates poor performance. In this case, 0.675 demonstrates a relatively high level of accuracy and reliability in the model's predictions.

## Python Implementation

### Code Snippet

```python
import math
from typing import Union

def calculate_mcc(tp: int, tn: int, fp: int, fn: int) -> Union[float, None]:
    """
    Calculate Matthew's Correlation Coefficient (MCC).

    Parameters:
    tp (int): True Positives
    tn (int): True Negatives
    fp (int): False Positives
    fn (int): False Negatives

    Returns:
    float: MCC value or None if the calculation is invalid
    """
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return None  # Return None if denominator is zero to avoid division by zero

    return numerator / denominator

# Example usage
tp = 50
tn = 30
fp = 10
fn = 5
mcc = calculate_mcc(tp, tn, fp, fn)
print(f"MCC: {mcc}")
```

### Explanation

**Function Definition**: The function `calculate_mcc` is defined with parameters for true positives (tp), true negatives (tn), false positives (fp), and false negatives (fn).

**Numerator Calculation**: The numerator of the MCC formula is calculated as $$(tp \times tn) - (fp \times fn)$$.

**Denominator Calculation**: The denominator is calculated as the square root of the product $$(tp + fp) \times (tp + fn) \times (tn + fp) \times (tn + fn)$$.

**Division by Zero Check**: Before performing the division, the function checks if the denominator is zero to avoid division by zero errors. If the denominator is zero, the function returns `None`.

**MCC Calculation**: If the denominator is not zero, the function returns the MCC value by dividing the numerator by the denominator.

## Interpretation of MCC

### Range

MCC values range from -1 to +1. This range allows for a detailed interpretation of the performance of a binary classification model.

### Significance of Values

- **+1**: A value of +1 indicates a perfect prediction, where the model correctly classifies all positive and negative instances.
- **0**: A value of 0 suggests that the model's predictions are no better than random chance. In other words, the model has no discriminative power.
- **-1**: A value of -1 indicates a completely wrong prediction, where the model's predictions are inversely related to the actual outcomes.

### Comparison

MCC provides a more comprehensive evaluation of binary classifiers compared to other metrics:

- **Accuracy**: While accuracy measures the proportion of correct predictions, it can be misleading, especially with imbalanced datasets. A model might have high accuracy by simply predicting the majority class most of the time.
- **F1 Score**: The F1 Score balances precision and recall, providing a better measure than accuracy for imbalanced datasets. However, it still doesn't account for true negatives, which can be important in certain contexts.
- **MCC**: By considering all four categories of the confusion matrix (TP, TN, FP, FN), MCC offers a balanced evaluation, even in the presence of class imbalance. It provides a single value that captures the overall performance of the model, making it a robust metric for various classification tasks.

## Advantages of MCC

### Balanced Measure

MCC is a balanced measure that considers all four categories of the confusion matrix: true positives, true negatives, false positives, and false negatives. This balanced nature ensures that the metric does not favor any particular class, providing a more holistic evaluation of a binary classification model's performance.

### Use in Imbalanced Datasets

MCC is particularly useful in scenarios involving imbalanced datasets. Traditional metrics like accuracy can be misleading in such cases, as they may give a high score simply by predicting the majority class correctly. MCC, however, accounts for both false positives and false negatives, providing a more accurate representation of a model's performance when the classes are of unequal sizes.

### Comprehensive Evaluation

Compared to other metrics, MCC offers a more comprehensive evaluation of a model's predictive capabilities. While metrics like accuracy and F1 score focus on specific aspects of performance, MCC encapsulates the overall effectiveness of a model. It provides a single value that reflects the balance between precision and recall, making it a robust and reliable metric for assessing binary classifiers.

## Use Cases

### Real-World Examples

MCC is particularly useful in various real-world scenarios where the balance between true positives, true negatives, false positives, and false negatives is critical. For instance, in medical diagnostics, correctly identifying both diseased and healthy individuals is crucial, making MCC an ideal metric for evaluating models.

### Case Studies

#### Healthcare

In the healthcare domain, MCC can be used to evaluate models that predict the presence of diseases. For example, in predicting the presence of cancer from medical images, a high MCC value would indicate that the model accurately distinguishes between malignant and benign cases. This is essential to ensure both high detection rates (minimizing false negatives) and low misdiagnosis rates (minimizing false positives).

#### Finance

In finance, MCC can be applied to models predicting credit defaults. Accurately identifying both good and bad credit risks is vital to minimize financial losses and maintain customer satisfaction. An MCC evaluation ensures that the model performs well across all risk categories, reducing the likelihood of approving risky loans or rejecting good customers.

#### Fraud Detection

MCC is also valuable in fraud detection systems, where it is important to correctly identify fraudulent transactions (true positives) while minimizing the false identification of legitimate transactions as fraudulent (false positives). This balance is crucial for maintaining security without disrupting customer experience.

### Applications in Different Domains

- **Healthcare**: Diagnosis prediction, disease outbreak detection, patient outcome prediction.
- **Finance**: Credit scoring, loan approval processes, fraud detection.
- **Retail**: Customer churn prediction, inventory management, personalized marketing.
- **Cybersecurity**: Intrusion detection systems, malware classification, phishing attack detection.

By using MCC, these applications can ensure a balanced and comprehensive evaluation of their predictive models, leading to more reliable and effective decision-making processes.

## Limitations

### Potential Drawbacks

While MCC is a robust metric, it does have certain limitations. One potential drawback is that MCC can be difficult to interpret intuitively compared to simpler metrics like accuracy or precision. Additionally, MCC requires the computation of all elements of the confusion matrix, which might not be straightforward in some contexts.

### Alternative Scenarios

There are scenarios where MCC might not be the best choice:

- **Multi-Class Classification**: MCC is primarily designed for binary classification. For multi-class problems, other metrics like the macro-averaged F1 score or the Cohen's Kappa coefficient might be more appropriate.
- **Interpretability**: In cases where stakeholders need easily interpretable metrics, simpler measures like accuracy or the F1 score could be preferred, despite their potential shortcomings in certain contexts.
- **Class Imbalance Handling**: Although MCC handles class imbalance well, in extremely imbalanced datasets, more specialized metrics like the balanced accuracy or area under the precision-recall curve (AUC-PR) might offer more insights.

By understanding these limitations, practitioners can make informed decisions on when to use MCC and when alternative metrics might be more suitable.

## Conclusion

### Summary

In this article, we explored Matthew’s Correlation Coefficient (MCC) as a measure for evaluating binary classification models. We discussed its definition, historical background, and significance in machine learning and statistics. We also delved into the mathematical formulation of MCC, providing a step-by-step example calculation. Furthermore, we highlighted the advantages of MCC, particularly its balanced nature and suitability for imbalanced datasets, and presented real-world use cases across various domains. Lastly, we considered the limitations of MCC and scenarios where alternative metrics might be more appropriate.

### Final Thoughts

MCC stands out as a powerful and comprehensive metric for assessing binary classifiers. Its ability to consider all four components of the confusion matrix ensures a balanced evaluation, making it particularly useful in cases of class imbalance. By incorporating MCC into the evaluation process, data scientists and practitioners can achieve a more accurate and reliable understanding of their models' performance. As such, MCC is an essential tool for ensuring robust and effective classification outcomes.

## References

1. Matthews, B. W. (1975). Comparison of the predicted and observed secondary structure of T4 phage lysozyme. *Biochimica et Biophysica Acta (BBA) - Protein Structure*, 405(2), 442-451. doi:10.1016/0005-2795(75)90109-9

2. Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21, 6. doi:10.1186/s12864-019-6413-7

3. Powers, D. M. W. (2011). Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation. *Journal of Machine Learning Technologies*, 2(1), 37-63.

4. Baldi, P., Brunak, S., Chauvin, Y., Andersen, C. A. F., & Nielsen, H. (2000). Assessing the accuracy of prediction algorithms for classification: An overview. *Bioinformatics*, 16(5), 412-424. doi:10.1093/bioinformatics/16.5.412

5. Tharwat, A. (2018). Classification assessment methods. *Applied Computing and Informatics*, 17(1), 168-192. doi:10.1016/j.aci.2018.08.003

6. Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861-874. doi:10.1016/j.patrec.2005.10.010

7. Hand, D. J., & Till, R. J. (2001). A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems. *Machine Learning*, 45(2), 171-186. doi:10.1023/A:1010920819831

## Appendix A - Fortran Code Snippet

```fortran
program mcc_calculation
    implicit none

    ! Variable declarations
    integer :: tp, tn, fp, fn
    real :: mcc

    ! Assign hypothetical values to the confusion matrix components
    tp = 50
    tn = 30
    fp = 10
    fn = 5

    ! Calculate MCC
    mcc = calculate_mcc(tp, tn, fp, fn)

    ! Print the MCC result
    if (mcc /= -2.0) then
        print *, 'MCC: ', mcc
    else
        print *, 'MCC calculation is invalid (division by zero).'
    endif

contains

    ! Function to calculate MCC
    real function calculate_mcc(tp, tn, fp, fn)
        implicit none
        integer, intent(in) :: tp, tn, fp, fn
        real :: numerator, denominator

        ! Calculate numerator and denominator
        numerator = real(tp * tn - fp * fn)
        denominator = sqrt(real((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

        ! Check for division by zero
        if (denominator == 0.0) then
            calculate_mcc = -2.0
        else
            calculate_mcc = numerator / denominator
        endif
    end function calculate_mcc

end program mcc_calculation
```

### Explanation

**Variable Declarations**: We declare the confusion matrix components `tp`, `tn`, `fp`, and `fn` as integers, and the resulting `mcc` as a real number.

**Assign Hypothetical Values**: Assign values to `tp`, `tn`, `fp`, and `fn` as per the example.

**Calculate MCC**: Call the `calculate_mcc` function to compute the MCC value.

**Print the MCC Result**: Print the MCC result. If the calculation is invalid (denominator is zero), print an error message.

**Function to Calculate MCC**:

- **Numerator Calculation**: Compute the numerator as $$(tp \times tn) - (fp \times fn)$$.
- **Denominator Calculation**: Compute the denominator as the square root of the product $$(tp + fp) \times (tp + fn) \times (tn + fp) \times (tn + fn)$$.
- **Division by Zero Check**: Check if the denominator is zero. If it is, return a special value (-2.0) indicating an invalid calculation. Otherwise, return the MCC value.

### Compilation and Execution

To compile and run this Fortran program, save it to a file (e.g., `mcc_calculation.f90`) and use a Fortran compiler like `gfortran`:

```sh
gfortran -o mcc_calculation mcc_calculation.f90
./mcc_calculation
```

## Appendix B: C Code Snippet

```c

#include <stdio.h>
#include <math.h>

float calculate_mcc(int tp, int tn, int fp, int fn) {
    float numerator, denominator;

    // Calculate numerator and denominator
    numerator = (float)(tp * tn - fp * fn);
    denominator = sqrt((float)(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));

    // Check for division by zero
    if (denominator == 0.0) {
        return -2.0; // Return a special value indicating invalid calculation
    } else {
        return numerator / denominator;
    }
}

int main() {
    // Variable declarations
    int tp = 50, tn = 30, fp = 10, fn = 5;
    float mcc;

    // Calculate MCC
    mcc = calculate_mcc(tp, tn, fp, fn);

    // Print the MCC result
    if (mcc != -2.0) {
        printf("MCC: %f\n", mcc);
    } else {
        printf("MCC calculation is invalid (division by zero).\n");
    }

    return 0;
}
```
