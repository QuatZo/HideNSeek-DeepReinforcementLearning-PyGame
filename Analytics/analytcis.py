import pandas as pd
import os
import json
from collections.abc import Iterable
import matplotlib.pyplot as plt
import statsmodels

import plotly.graph_objects as go
import plotly as py
import plotly.express as px


def read_json(path_to_json = 'Inż/'):
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

def parse_json_2_dataframe(jsons,unwanted_col = 'unwanted_col'):
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
                    col = pd.Series(json[key],name=key)
                    list_of_cols.append(col)
                else:
                    dataframe_aborted_cols.append(pd.Series(json[key],name=key))

        dataframe = pd.DataFrame(list_of_cols).T
        cleaned_dataframe = dataframe[['timestamps', 'episode_lengths', 'episode_rewards',
           'episode_winners', 'episode_types']].dropna(axis=0)
        config_and_best = dataframe[['episode_best','config']]

        config = config_and_best.transpose()[['game', 'hiding', 'seeker', 'video']].drop('episode_best', axis=0)


        for params in [config[col] for col in config.columns]:
                for val in params.values:
                    df = pd.DataFrame(list(val.items())).set_index(0).transpose()
                    if params.name == 'hiding':
                        hiding_df = df
                    elif params.name == 'seeker':
                        seeker_df = df
                        seeker_df = seeker_df.join(hiding_df,rsuffix='_hiding',lsuffix='_seeker')
                    elif params.name == 'video':
                        video_df = df
                    elif params.name == 'game':
                        game_df = df
        seeker_df = seeker_df.merge(game_df,left_index=True, right_index=True)
        param_config_df = seeker_df.merge(video_df,left_index=True, right_index=True)
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

def plot_rewards(data,agent):

    df = data.copy()
    df.episode_rewards = [parse_episode_rewards(episode_reward) for episode_reward in zip(df.episode_rewards)]
    agents_rewards = df.episode_rewards.apply(pd.Series)
    df['seeker_rewards'] = agents_rewards[0]
    df['hiding_rewards'] = agents_rewards[1]
    df.drop('episode_rewards',axis=1,inplace=True)
    df["episode_winners"] = df["episode_winners"].astype('category')
    df["episode_winners_cat"] = df["episode_winners"].cat.codes

    m = df['episode_winners_cat'].astype(bool)
    n = ~m
    df['seeker_win'] = (
        m.groupby([m, (~m).cumsum().where(m)]).cumcount().add(1).mul(m))
    df['hiding_win'] = (
        m.groupby([n, (~n).cumsum().where(n)]).cumcount().add(1).mul(n))
    df['consecutive_result'] = df[['hiding_win','seeker_win']].max(axis=1)
    df.drop('timestamps',inplace=True,axis=1)

    agent_data = df[df['episode_winners'] == agent].copy()
    agent_data = agent_data[agent_data['consecutive_result'] >= agent_data['consecutive_result'].mean()]
    agent_data['std_hiding_rewards'] = agent_data['hiding_rewards'].std()
    agent_data['std_seeker_rewards'] = agent_data['seeker_rewards'].std()
    print(agent_data.index )
    fig = px.scatter(agent_data, x=agent_data.index, y=['seeker_rewards', 'hiding_rewards'], width=600, height=400,
                     size='consecutive_result',
                     labels=dict(index="Episode", value="Amount of Points", variable='Colors',
                                 consecutive_result='Consecutive_result', episode_winners='Episode_winners'),title=str("winner: " + agent),
                     hover_data=['episode_winners','std_seeker_rewards','std_hiding_rewards'],trendline="lowess")
    fig.show()
    return fig.to_html()



def data_to_excel(data,param_config_data,path):
    """
           Reads jsons in the specified dir
            Parameters
            ----------
                data : pd.DataFrame     output of parse_json_2_dataframe

                param_config_data :  pd.DataFrame   output of parse_json_2_dataframe

                path : str    path and filename for output .xlsx

            Returns
            -------
                sum(seeker_sum) : int   sum of seeker rewards

                sum(hiding_sum) : int      sum of hiding rewards
        """
    df = data.copy()
    param_config_df = param_config_data
    df.episode_rewards = [parse_episode_rewards(episode_reward) for episode_reward in zip(df.episode_rewards)]
    agents_rewards = df.episode_rewards.apply(pd.Series)
    df['seeker_rewards'] = agents_rewards[0]
    df['hiding_rewards'] = agents_rewards[1]
    df.drop('episode_rewards',axis=1,inplace=True)
    df["episode_winners"] = df["episode_winners"].astype('category')
    df["episode_winners_cat"] = df["episode_winners"].cat.codes

    m = df['episode_winners_cat'].astype(bool)
    n = ~m
    df['seeker_win'] = (
        m.groupby([m, (~m).cumsum().where(m)]).cumcount().add(1).mul(m))
    df['hiding_win'] = (
        m.groupby([n, (~n).cumsum().where(n)]).cumcount().add(1).mul(n))
    df['consecutive_result'] = df[['hiding_win','seeker_win']].max(axis=1)
    df.drop('timestamps',inplace=True,axis=1)

    df_with_params = df.append(param_config_df, ignore_index=True)
    for col in param_config_df.columns:
        df_with_params[col].bfill(inplace=True)
    df_with_params[:-1].to_excel(path)



def summary_of_traings():
    for training in os.listdir('Inż/'):
        path_to_training = 'Inż/' + training + '/'
        for core in os.listdir(path_to_training):
            path_to_core = path_to_training + core + '/'
            data, param_config_df = parse_json_2_dataframe(read_json(path_to_core))
            output = path_to_core + 'data.xlsx'
            data_to_excel(data,param_config_df,output)

def merge_excels():


    data = []
    for training in os.listdir('Inż/'):
        path_to_training = 'Inż/' + training + '/'
        for core in os.listdir(path_to_training):
            path_to_core = path_to_training + core + '/'
            for file_name in [file for file in os.listdir(path_to_core) if file.endswith('data.xlsx')]:
                with open(path_to_core + file_name,'rb') as unexcellent_excel:
                    df = pd.read_excel(unexcellent_excel)
                    data.append(df)


    excellent_excel = pd.concat(data, axis=0)
    excellent_excel.to_excel('excellent_excel.xlsx')


# merge_excels()
data, param_config_df = parse_json_2_dataframe(read_json())
plot_rewards(data,'HIDING')
# data_to_excel(data,param_config_df,'danetrololo.xlsx')



