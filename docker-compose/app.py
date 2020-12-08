from flask import Flask, render_template, jsonify, request
from celery import Celery
from celery.result import AsyncResult

import time
import copy
import datetime
from pytz import timezone
from pathlib import Path
import statistics

import os
import json
import pandas as pd
import plotly.io as pltio
import plotly.express as px
import statsmodels

import game_env.hidenseek_gym
from game_env.hidenseek_gym.config import config as default_config

from helpers import Helpers

app = Flask(__name__)
celery = Celery(broker='redis://redis:6379/0', backend='redis://redis:6379/0')
celery.conf.broker_transport_options = {
    "visibility_timeout": 3600 * 24 * 360}  # 1h * 24 * 360 = 360d


def read_json(path_to_json='data/input/'):
    """
    Reads jsons in the specified dir

    Parameters
    ----------
        path_to_json : string


    Returns
    -------
        List of jsons.
    """
    data = []
    for file_name in [file for file in os.listdir(path_to_json) if file.endswith('openaigym.episode_batch.stats.json')]:
        with open(path_to_json + file_name) as json_file:
            data.append(json.load(json_file))
            return data


def parse_json_2_dataframe(jsons, unwanted_col='unwanted_col'):
    """
    Reads jsons in the specified dir

    Parameters
    ----------
        jsons : List of jsons.

        list_of_cols : list of columns from which DataFrame is made.

        unwanted_col : OPTIONAL, str.

        dataframe_aborted_cols : OPTIONAL, list of aborted columns



    Returns
    -------
        cleaned_dataframe : pd.DataFrame    N long table, N is number of episodes in the data.

        param_config_df : pd.DataFrame      17x1 table with parameter values
    """
    for json in jsons:
        list_of_cols = []
        dataframe_aborted_cols = []
        for key in json:
            if not isinstance(json[key], float):
                if key != unwanted_col:
                    col = pd.Series(json[key], name=key)
                    list_of_cols.append(col)
                else:
                    dataframe_aborted_cols.append(
                        pd.Series(json[key], name=key))

        dataframe = pd.DataFrame(list_of_cols).T
        cleaned_dataframe = dataframe[['timestamps', 'episode_lengths', 'episode_rewards',
                                       'episode_winners', 'episode_types']].dropna(axis=0)
        config_and_best = dataframe[['episode_best', 'config']]

        config = config_and_best.transpose(
        )[['game', 'hiding', 'seeker', 'video']].drop('episode_best', axis=0)

        for params in [config[col] for col in config.columns]:
            for val in params.values:
                df = pd.DataFrame(list(val.items())).set_index(0).transpose()
                if params.name == 'hiding':
                    hiding_df = df
                elif params.name == 'seeker':
                    seeker_df = df
                    seeker_df = seeker_df.join(
                        hiding_df, rsuffix='_hiding', lsuffix='_seeker')
                elif params.name == 'video':
                    video_df = df
                elif params.name == 'game':
                    game_df = df
        seeker_df = seeker_df.merge(game_df, left_index=True, right_index=True)
        param_config_df = seeker_df.merge(
            video_df, left_index=True, right_index=True)
        return cleaned_dataframe, param_config_df


def parse_episode_rewards(episode_reward):
    """
    Reads jsons in the specified dir

    Parameters
    ----------
        episode_reward : pd.Series  containing list of lists

    Returns
    -------
        sum(seeker_sum) : int   sum of seeker rewards

        sum(hiding_sum) : int      sum of hiding rewards
    """
    seeker_sum, hiding_sum = [], []
    for frame in episode_reward:
        for seeker_reward, hiding_reward in frame:
            seeker_sum.append(seeker_reward)
            hiding_sum.append(hiding_reward)
    return sum(seeker_sum), sum(hiding_sum)


def plot_rewards(data, agent, path_to_save):
    df = data.copy()
    df.episode_rewards = [parse_episode_rewards(
        episode_reward) for episode_reward in zip(df.episode_rewards)]
    agents_rewards = df.episode_rewards.apply(pd.Series)
    df['seeker_rewards'] = agents_rewards[0]
    df['hiding_rewards'] = agents_rewards[1]
    df.drop('episode_rewards', axis=1, inplace=True)
    df["episode_winners"] = df["episode_winners"].astype('category')
    df["episode_winners_cat"] = df["episode_winners"].cat.codes

    m = df['episode_winners_cat'].astype(bool)
    n = ~m
    df['seeker_win'] = (
        m.groupby([m, (~m).cumsum().where(m)]).cumcount().add(1).mul(m))
    df['hiding_win'] = (
        m.groupby([n, (~n).cumsum().where(n)]).cumcount().add(1).mul(n))
    df['consecutive_result'] = df[['hiding_win', 'seeker_win']].max(axis=1)
    df.drop('timestamps', inplace=True, axis=1)

    agent_data = df[df['episode_winners'] == agent].copy()
    agent_data = agent_data[agent_data['consecutive_result']
                            >= agent_data['consecutive_result'].mean()]
    agent_data['std_hiding_rewards'] = agent_data['hiding_rewards'].std()
    agent_data['std_seeker_rewards'] = agent_data['seeker_rewards'].std()

    if len(agent_data.index) != 0:
        fig = px.scatter(agent_data, x=agent_data.index, y=['seeker_rewards', 'hiding_rewards'], width=600, height=400,
                         size='consecutive_result',
                         labels=dict(index="Episode", value="Amount of Points", variable='Colors',
                                     consecutive_result='Consecutive_result', episode_winners='Episode_winners'), title=str("winner: " + agent),
                         hover_data=['episode_winners', 'std_seeker_rewards', 'std_hiding_rewards'], trendline="lowess")
        with open(os.path.join(path_to_save, agent+'_wins_plot.html'), 'w+') as f:
            pltio.write_html(fig, f, auto_open=False)

        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            width=500,
        )

        return fig.to_html(include_plotlyjs=False, full_html=False)

    else:
        text = 'No winnings for ' + agent.lower() + ' agent'
        return "<div class='display-2'> " + text + " </div>"


@celery.task(name='train.core', bind=True)
def train(self, core_id, config_data, start_date):
    start = time.time()
    cfg = Helpers.prepare_config(config_data)

    walls, seeker, hiding, width, height = Helpers.prepare_map(cfg)

    env, step_img_path, fps_batch, render_mode, wins_l = Helpers.create_env(
        config=cfg,
        width=width,
        height=height,
        walls=walls,
        seeker=seeker,
        hiding=hiding,
        start_date=start_date,
        core_id=core_id,
    )

    AGENTS = 2
    algorithm = Helpers.pick_algorithm(cfg, env=env, agents=AGENTS)
    algorithm.prepare_model()

    for i in range(cfg['game']['episodes']):
        algorithm.before_episode()
        metadata = Helpers.update_celery_metadata(
            core_id=core_id,
            curr=i + 1,
            total=cfg['game']['episodes'],
            ep_iter=cfg['game']['duration'],
            fps=None,
            itera=0,
            iter_perc=0,
            time_elap=round(time.time() - start),
            img_path=step_img_path,
            eta=None,
            rewards=[0, 0],
            wins=[0, 0],
            wins_moving=[0, 0],
        )
        self.update_state(state='PROGRESS', meta=metadata)

        obs_n, reward_n, rewards_ep, done, fps_episode = Helpers.new_ep(env)

        while True:
            rewards_ep = [rewards_ep[0] + reward_n[0],
                          rewards_ep[1] + reward_n[1]]
            metadata['status'] = Helpers.update_metadata_status(
                fps=env.clock.get_fps(),
                itera=int(cfg['game']['duration']) - env.duration,
                iter_perc=round(
                    ((int(cfg['game']['duration']) - env.duration) / int(cfg['game']['duration'])) * 100, 2),
                time_elap=round(time.time() - start),
                eta=round((env.duration / env.clock.get_fps()) + int(cfg['game']['duration']) / env.clock.get_fps(
                ) * (int(cfg['game']['episodes']) - i)) if env.clock.get_fps() else None,
                img_path=step_img_path[8:],
                rewards=rewards_ep,
                wins=[sum(w) for w in wins_l],
                wins_moving=[sum(w[-10:]) for w in wins_l],
            )

            fps_episode.append(env.clock.get_fps())

            algorithm.before_action(obs_n=obs_n)

            action_n = algorithm.take_action(obs_n=obs_n)
            obs_old_n = copy.deepcopy(obs_n)

            algorithm.before_step(action_n=action_n)
            obs_n, reward_n, done, _ = env.step(action_n)
            algorithm.after_step(
                reward_n=reward_n,
                obs_old_n=obs_old_n,
                obs_n=obs_n,
                done=done
            )

            Helpers.update_img_status(
                env, cfg['video']['monitoring'], step_img_path, render_mode)
            self.update_state(state='PROGRESS', meta=metadata)

            if done[0]:
                algorithm.handle_gameover(
                    obs_n=obs_n,
                    reward_n=reward_n,
                    ep_length=int(cfg['game']['duration']) - env.duration,
                )
                Helpers.handle_gameover(done[1], wins_l)
                break

        algorithm.after_episode()

        fps_batch.append(statistics.fmean(fps_episode))

    algorithm.before_cleanup()
    Helpers.cleanup(env, core_id)

    metadata = {
        'step': 'read meta file'
    }
    self.update_state(state='PLOTTING', meta=metadata)
    path_to_file = '/opt/app/monitor/' + \
        start_date + "/core-" + str(core_id) + '/'
    raw_stats = read_json(path_to_file)

    metadata = {
        'step': 'parse meta file'
    }
    self.update_state(state='PLOTTING', meta=metadata)
    parsed_stats, _ = parse_json_2_dataframe(raw_stats)

    metadata = {
        'step': 'plot seeker',
    }
    self.update_state(state='PLOTTING', meta=metadata)
    seeker_plot = plot_rewards(parsed_stats, 'SEEKER', path_to_file)

    metadata = {
        'step': 'plot hiding',
    }
    self.update_state(state='PLOTTING', meta=metadata)
    hiding_plot = plot_rewards(parsed_stats, 'HIDING', path_to_file)

    return Helpers.get_celery_success(
        core_id=core_id,
        time_elap=round(time.time() - start, 4),
        fps_batch=fps_batch,
        wins=[sum(w) for w in wins_l],
        seeker_plot=seeker_plot,
        hiding_plot=hiding_plot,
    )


@app.route('/status/<task_id>')
def get_task_status(task_id):
    task = train.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending... Why tho'
        }
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'result': task.result,
        }
    elif task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', {}),
            'episode_iter': task.info.get('episode_iter', 0),
            'config': task.info.get('config', {}),
        }
    elif task.state == 'PLOTTING':
        response = {
            'state': task.state,
            'step': task.info.get('step', 0),
        }
    else:
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # exception
        }

    return jsonify(response)


@ app.route('/train', methods=['POST'])
def start_training():
    data = request.json

    tz_local = timezone('Europe/Warsaw')
    now = datetime.datetime.now(tz=tz_local)
    start_date = datetime.datetime.strftime(now, "%Y-%m-%dT%H-%M-%SZ")

    tasks = list()
    for i in range(int(data['cpus'])):
        Path('/opt/app/static/images/core-' +
             str(i)).mkdir(parents=True, exist_ok=True)
        task = train.apply_async((i, data['configs'][i], start_date))
        tasks.append(task.id)

    return {'task_ids': tasks, 'start_date': start_date}, 202


@ app.route('/')
def homepage():
    return render_template('homepage.html', cfg=default_config)


if __name__ == '__main__':
    celery.control.purge()
    app.run(debug=True, host='0.0.0.0')
