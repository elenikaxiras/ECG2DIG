(work in progress)

## ECG2DIG: Artificial intelligence-enabled electrocardiogram to detect digoxin exposure and toxicity

**Constantine Tarabanis, MD; Eleni Angelaki, PhD; Sam F. Friedman, PhD; Danielle Pace, PhD; Mahnaz Maddah, PhD; Hanspeter Pfister, PhD; Patrick T. Ellinor, MD, PhD; Shaan Khurshid, MD, MPH**

The code associated with this publication is covered under the GNU General Public License.

This repository is work in progress and when completed will provide a reproducible pipeline for training and evaluating ECG2DIG (ECG to Digoxin), an artificial intelligence (AI)-enabled 12-lead electrocardiogram (ECG) analysis model that may detect digoxin exposure and discriminate supratherapeutic levels. The model consists of a one-dimensional convolutional neural network (CNN), specialized for time-series analysis. Each of the 12 ECG leads is first passed through a fixed finite–impulse–response (FIR) band–pass filter to suppress baseline drift and high–frequency noise, and the 12 leads are then integrated with an enhanced attention mechanism that weighs more informative leads before aggregation. In keeping with prior ECG models demonstrating improved performance when incorporating related tasks, ECG2DIG was a multi-task model with a primary task of digoxin exposure classification, along with auxiliary tasks of heart rate estimation and regression of the digoxin level. To promote generalization, we applied standard architectural regularization (batch normalization and dropout) and lightweight data augmentation. 

-----------------------------------------------------------------------------------------------------------------------
### Citation

Citation information (journal link) -- coming soon

-----------------------------------------------------------------------------------------------------------------------
### Model Architecture

![ECG2DIG_highlevel_sk.pdf](https://github.com/user-attachments/files/23604037/ECG2DIG_highlevel_sk.pdf)

-----------------------------------------------------------------------------------------------------------------------
### Code structure

├── data<BR>
├── inference.ipynb<BR>
├── models<BR>
│   └── ECG2DIG.py<BR>
├── LICENSE<BR>
├── README.md<BR>
└── src<BR>

