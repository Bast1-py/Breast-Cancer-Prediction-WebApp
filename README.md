INFO:__main__:Training and Comparing Candidate Models (First Run)...
INFO:__main__:Random Forest Model test accuracy: 96.330%
INFO:__main__:Support Vector Machine (SVM) Model test accuracy: 89.908%
INFO:__main__:Decision Tree Model test accuracy: 93.578%
C:\Users\Bastian Torus\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\ensemble\_weight_boosting.py:519: FutureWarning: The SAMME.R algorithm (the default) is deprecated and will be removed in 1.6. Use the SAMME algorithm to circumvent this warning.
  warnings.warn(
INFO:__main__:AdaBoost Model  test accuracy: 92.661%
INFO:__main__:CatBoost Model  test accuracy: 97.248%
INFO:__main__:HistGradientBoosting test accuracy: 97.248%
INFO:__main__:Best Model: CatBoost Model (97.248%) - Deploying This One.
 * Serving Flask app 'app'
INFO:werkzeug:WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
INFO:werkzeug:Press CTRL+C to quit
