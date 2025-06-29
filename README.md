
# SSVEP Machine Learning Visualization

This repository contains Python scripts and sample data visualizations used in the study:
**"Comparative Analysis of Machine Learning Algorithms for EEG-Based SSVEP Classification Using Consumer-Grade Hardware"**.

## Contents

- `visualization.py`: Python script to generate visualizations of classification performance.
- `figures/`: Output images including:
  - Classification accuracy bar chart
  - Multi-metric radar plot
  - Simulated cross-validation boxplot

## How to Use

1. Make sure you have Python 3.x and pip installed.
2. Install required libraries using:
   ```
   pip install -r requirements.txt
   ```
3. Run the script:
   ```
   python visualization.py
   ```

## Output

Images will be saved in the `figures/` folder:
- `ssvep_accuracy.png`
- `ssvep_radar.png`
- `ssvep_cv_stability.png`

## License

This project is licensed under the MIT License.
