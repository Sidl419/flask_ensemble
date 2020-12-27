import os
import time
import ensembles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from werkzeug.utils import secure_filename
from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, FileField, SelectField, FloatField, DecimalField

import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


app = Flask(__name__, template_folder='html')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'img')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
data_path = './../data'
Bootstrap(app)
models = []


model_names = [('1', 'Random Forest'), ('2', 'Gradient Boosting')]

class ModelForm(FlaskForm):
    model = SelectField('Model', choices=model_names)

    n_estimators = DecimalField('Number of trees', validators=[DataRequired()])
    feature_subsample_size = DecimalField('Feature subsample size', validators=[DataRequired()])
    max_depth = DecimalField('Max depth of one tree', validators=[DataRequired()])
    learning_rate = DecimalField('Learning rate of one tree (any for random forest)', validators=[DataRequired()])

    data_path = FileField('Train data', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])

    target_path = FileField('Train target', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])

    submit = SubmitField('Train model')

class TrainForm(ModelForm):
    data_path = StringField('Train data', validators=[DataRequired()])
    target_path = StringField('Train target', validators=[DataRequired()])

    time = StringField('Training time', validators=[DataRequired()])

    submit = SubmitField('Next step')


class Response(FlaskForm):
    score = StringField('RMSE', validators=[DataRequired()])
    new_model = SubmitField('New model')
    new_data = SubmitField('New test data')

class PredictForm(FlaskForm):
    data_path = FileField('Test data', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])

    target_path = FileField('Test target', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])

    submit = SubmitField('Get result')


def read_data(form, target, log_name='Data'):
    if target:
        lines = form.target_path.data.stream.readlines()
        f = form.target_path.data
    else:
        lines = form.data_path.data.stream.readlines()
        f = form.data_path.data

    app.logger.info(f'{log_name}: uploaded {len(lines)} lines')

    filename = secure_filename(f.filename)
    datapath = os.path.join(data_path, filename)
    with open(datapath, 'wb') as f_out:
        for l in lines:
            f_out.write(l)
    return datapath, filename


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r   

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def get_result():
    try:
        response_form = Response()

        if response_form.validate_on_submit():
            if response_form.new_model.data:
                models.pop()
                return redirect(url_for('get_model'))
            else:
                return redirect(url_for('get_predict'))

        response_form.score.data = request.args.get('score')

        return render_template('from_form.html', form=response_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

@app.route('/predict', methods=['GET', 'POST'])
def get_predict():
    try:
        predict_form = PredictForm()

        if request.method == 'POST' and predict_form.validate_on_submit():
            #os.remove(os.path.join('static/img', str(request.args.get('filename'))))

            datapath, _ = read_data(predict_form, target=False, log_name='Test data')
            targetpath, _ = read_data(predict_form, target=True, log_name='Test target')

            test = pd.read_csv(datapath, index_col=0).to_numpy()
            true_y = pd.read_csv(targetpath).to_numpy().reshape((-1))

            app.logger.info('Test data shape: {}, {}'.format(test.shape, true_y.shape))

            pred = models[-1].predict(test)
            score = np.power(pred - true_y, 2).mean()**(1/2)

            os.remove(datapath)
            os.remove(targetpath)

            app.logger.info('Score: {}'.format(score))

            return redirect(url_for('get_result', score=score))

        return render_template('from_form.html', form=predict_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

@app.route('/train', methods=['GET', 'POST'])
def train_results():
    try:
        model_form = TrainForm()

        if model_form.validate_on_submit():
            return redirect(url_for('get_predict'))

        model_type = request.args.get('modelform')
        model_form.model.data = model_type
        iter_count = len(models[-1].history)
        if model_type == '1':
            iters = np.arange(iter_count)
            model_form.learning_rate.data = -1
        else:
            iters = np.arange(1, iter_count + 1)
            model_form.learning_rate.data = models[-1].learning_rate
        
        model_form.n_estimators.data = models[-1].n_estimators
        model_form.feature_subsample_size.data = models[-1].feature_subsample_size
        model_form.max_depth.data = models[-1].max_depth

        model_form.data_path.data = request.args.get('dataform')
        model_form.target_path.data = request.args.get('targetform')

        model_form.time.data = request.args.get('train_time') + 's'

        plt.figure(figsize=(10, 7))
        plt.plot(iters, models[-1].history)
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.title('Loss over iterations')
        filename = 'loss.jpg'
        plt.savefig(os.path.join('static/img', filename))

        return render_template('train_res.html', form=model_form, filename=filename)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

@app.route('/model', methods=['GET', 'POST'])
def get_model():
    try:
        model_form = ModelForm()

        if request.method == 'POST' and model_form.validate_on_submit():
            app.logger.info('Model: {0}'.format(dict(model_names).get(model_form.model.data)))
            app.logger.info('n_estimators: {}, feature_subsample_size: {}, max_depth: {}, learning_rate: {}'.format(
            model_form.n_estimators.data, model_form.feature_subsample_size.data, 
            model_form.max_depth.data, model_form.learning_rate.data))

            models.clear()

            if model_form.model.data == '1':
                model = ensembles.RandomForestMSE(n_estimators=int(model_form.n_estimators.data), 
                feature_subsample_size=float(model_form.feature_subsample_size.data), 
                max_depth=int(model_form.max_depth.data))
            else:
                model = ensembles.GradientBoostingMSE(n_estimators=int(model_form.n_estimators.data), 
                feature_subsample_size=float(model_form.feature_subsample_size.data), 
                max_depth=int(model_form.max_depth.data), learning_rate=float(model_form.learning_rate.data))

            datapath, dataname = read_data(model_form, target=False, log_name='Train data')
            targetpath, targetname = read_data(model_form, target=True, log_name='Train target')

            train = pd.read_csv(datapath, index_col=0).to_numpy()
            target = pd.read_csv(targetpath).to_numpy().reshape((-1))

            app.logger.info('train data shape: {}, {}'.format(train.shape, target.shape))

            start = time.time()
            model.fit(train, target)
            models.append(model)

            os.remove(datapath)
            os.remove(targetpath)

            train_time = time.time() - start
            app.logger.info('Training time: {}'.format(train_time))
            
            return redirect(url_for('train_results', dataform=dataname, 
                        targetform=targetname, modelform=model_form.model.data, train_time=round(train_time, 2)))
        return render_template('from_form.html', form=model_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
