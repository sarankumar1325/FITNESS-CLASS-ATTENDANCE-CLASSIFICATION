# Fitness Class Attendance Predictor

A Streamlit web application that predicts fitness class attendance using machine learning.



https://github.com/user-attachments/assets/bcd0e53f-1e42-40db-bba7-95acf3c9f475



## Features
- Attendance prediction with probability scores
- Interactive dashboard with visualizations
- Feature importance analysis
- Attendance trends visualization
- User-friendly interface

## Installation

```bash
git clone https://github.com/yourusername/fitness-predictor
cd fitness-predictor
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- streamlit
- pandas
- numpy
- plotly
- scikit-learn

## Usage

1. Place your trained model file as `final_model.pkl` in the project directory
2. Run the app:
```bash
streamlit run app.py
```
3. Access the app at `http://localhost:8501`

## Model Requirements
- Trained classifier with `predict` and `predict_proba` methods
- Feature importance attribute (optional)
- Input features:
  - months_as_member (int)
  - weight (float)
  - days_before (int)
  - day_of_week (categorical)
  - time (AM/PM)
  - category (class type)

## Directory Structure
```
fitness-predictor/
├── app.py
├── final_model.pkl
├── requirements.txt
└── README.md
```

## Contributing
Pull requests welcome. For major changes, open an issue first.

## License
MIT
