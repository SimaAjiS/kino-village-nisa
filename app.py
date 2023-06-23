import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

sns.set(style='darkgrid', font='Hiragino Maru Gothic Pro')

# 初期設定
st.set_page_config(page_title='積立NISAダッシュボードv100beta1', layout="wide")


# CSVファイルをデータフレームに整形加工
def format_csv2df(csv_file):
    # CSV読み込み
    cols = ['受渡日', 'ファンド名', '数量［口］', '単価', '受渡金額/(ポイント利用)[円]']
    df = pd.read_csv(csv_file, encoding='shift-jis', parse_dates=['受渡日'], usecols=cols)

    # str型をint型に変換 &ポイント利用(100)などの表記削除
    df['数量［口］'] = df['数量［口］'].str.replace(',', '').astype(int)
    df['単価'] = df['単価'].str.replace(',', '').astype(int)
    df['受渡金額/(ポイント利用)[円]'] = df['受渡金額/(ポイント利用)[円]'].str.replace('\(\d+\)', '', regex=True)
    df['受渡金額/(ポイント利用)[円]'] = df['受渡金額/(ポイント利用)[円]'].str.replace(',', '').astype(int)

    # ファンド名の表記ゆれ修正
    d = {'楽天・全世界株式インデックス・ファンド（楽天・バンガード・ファンド（全世界株式））': '楽天・全世界株式',
         '楽天・全世界株式インデックス・ファンド(楽天・VT)': '楽天・全世界株式',
         '楽天・全米株式インデックス・ファンド（楽天・バンガード・ファンド（全米株式））': '楽天・全米株式',
         '楽天・全米株式インデックス・ファンド(楽天・VTI)': '楽天・全米株式',
         '三井住友・DC年金バランス30(債券重点型)(マイパッケージ)': '三井住友・DC年金バランス30',
         'eMAXIS Slim 米国株式(S&P500)': 'eMAXIS Slim 米国株式(S&P500)',
         'eMAXIS Slim 新興国株式インデックス': 'eMAXIS Slim 新興国株式',
         'eMAXIS Slim 全世界株式(オール・カントリー)': 'eMAXIS Slim 全世界株式'
         }
    df['ファンド名'] = df['ファンド名'].replace(d)

    # 後の集計用に年月でまとめる
    df['受渡年月'] = df['受渡日'].dt.to_period('M')
    df = df[['受渡日', '受渡年月', 'ファンド名', '数量［口］', '単価', '受渡金額/(ポイント利用)[円]']]

    # カラム名簡易化
    df.columns = ['受渡日', '受渡年月', 'ファンド名', '数量', '単価', '受渡金額']

    return df


def calc_profit(df):
    # 期間中のファンド毎の所持数量（口数）と総受渡金額(ファンド別総積立額)
    df_by_found = df[['ファンド名', '数量', '受渡金額']].groupby('ファンド名').sum()

    # 期間最終月を取得する
    latest_month = df['受渡年月'].unique()[-1].strftime('%Y-%m')
    df_latest_month = df[df['受渡年月'] == latest_month][['ファンド名', '単価']]

    # df_by_foundをdf_latest_monthにマージ
    df_latest = pd.merge(df_latest_month, df_by_found, on='ファンド名')

    # カラム名を分かりやすく変更
    df_latest.columns = ['ファンド名', '最終月単価(円/1万口)', '所持数(口)', '受渡金額(円)']

    # 最新単価 ✕ 所持数量から資産額
    df_latest['資産価値(円)'] = (df_latest['最終月単価(円/1万口)'] / 10000) * df_latest['所持数(口)']

    # 資産価値 - 受渡金額よりファンド毎の損益計算
    df_latest['損益額(円)'] = df_latest['資産価値(円)'] - df_latest['受渡金額(円)']
    df_latest['損益率(％)'] = round(
        ((df_latest['資産価値(円)'] - df_latest['受渡金額(円)']) / df_latest['受渡金額(円)']) * 100, 2)

    # 期間中の総資産価値
    total_reserve = df_latest['受渡金額(円)'].sum()
    total_asset_value = df_latest['資産価値(円)'].sum()
    total_profit_amount = int(df_latest['損益額(円)'].sum())
    total_profit_rate = round(((total_asset_value - total_reserve) / total_reserve) * 100, 2)

    results = total_reserve, total_asset_value, total_profit_amount, total_profit_rate

    return df_latest, results


def place_metrics(df):
    # 3列表示
    col1, col2, col3 = st.columns(3)
    for index, row in df.iterrows():

        def create_metrics(row):

            reserve = row['受渡金額(円)']
            asset_value = row['資産価値(円)']
            profit_amount = int(row['損益額(円)'])
            profit_rate = round(((asset_value - reserve) / reserve) * 100, 2)

            # 指標表示
            st.metric(label=row['ファンド名'],
                      value=f'{profit_amount :+0d} 円',
                      delta=f'{profit_rate :+.2f} %')

        match index % 3:
            case 0:
                with col1:
                    create_metrics(row)
            case 1:
                with col2:
                    create_metrics(row)
            case 2:
                with col3:
                    create_metrics(row)


def place_pie(df_latest, results):
    labels = []
    asset_values = []

    for i, row in df_latest.iterrows():
        labels.append(row['ファンド名'])
        asset_values.append(row['資産価値(円)'])

    fig = plt.figure(figsize=(6, 4))
    plt.pie(
        x=asset_values,
        labels=labels,
        counterclock=False,
        startangle=90,
        pctdistance=0.7,
        autopct='%.1f %%',
        textprops={'fontsize': 8}
    )
    plt.pie([100], colors='white', radius=0.5)

    # 総資産価値と損益額、損益率
    total_asset_value = results[1] / 10000
    total_profit_amount = results[2] / 10000
    total_profit_rate = results[3]

    # ドーナツ中心にテキスト配置
    plt.text(x=0, y=0.2, s='積立額', ha='center', va='center', fontsize=10)
    plt.text(x=0, y=0.0, s=f'{total_asset_value :.1f}万円', ha='center', va='center', fontsize=16)
    plt.text(x=0, y=-0.2, s=f'{total_profit_amount :+.1f}万円', ha='center', va='center', fontsize=8)
    plt.text(x=0, y=-0.3, s=f'{total_profit_rate :+.2f}%', ha='center', va='center', fontsize=8)

    st.pyplot(fig)


def place_trend(df):
    fig = plt.figure(figsize=(12, 7))

    sns.lineplot(df.sort_values('単価', ascending=False),
                 x='受渡日', y='単価',
                 hue='ファンド名', style='ファンド名',
                 linewidth=3, palette='deep'
                 )

    plt.ylim(df['単価'].min(), df['単価'].max() * 1.3)

    st.pyplot(fig)


def place_progress_col(df_latest):
    # 損益率(％)を左側に配置する
    df_latest = df_latest[[
        'ファンド名',
        '損益率(％)',
        '最終月単価(円/1万口)',
        '所持数(口)',
        '受渡金額(円)',
        '資産価値(円)',
        '損益額(円)'
    ]]

    # Streamlit v1.23.1以降の新機能
    st.dataframe(
        data=df_latest,
        column_config={
            '損益率(％)': st.column_config.ProgressColumn(
                label='損益率(％)',
                format="+%f",
                min_value=-30,
                max_value=30,
            ),
        },
        hide_index=True
    )


def place_boxplt(df):
    fig = plt.figure()

    sns.boxplot(df.sort_values('単価', ascending=False),
                x='単価', y='ファンド名',
                width=0.6
                )

    # plt.xlim(0, 25000)
    st.pyplot(fig)


def main():
    # サイドバーでCSVファイル読み込み
    input = st.sidebar.file_uploader("CSVファイルを選択", type=['csv'], accept_multiple_files=False)

    if input is not None:
        df = format_csv2df(input)
        df_latest, results = calc_profit(df)

        if df is not None:
            col_metrics, col_graph = st.columns([3, 4])

            with col_metrics:
                # ドーナツグラフ配置
                place_pie(df_latest, results)

                st.markdown('---')

                # 指標を2行✕3列表示
                place_metrics(df_latest)

                st.markdown('---')

                # データフレーム
                place_progress_col(df_latest)

            with col_graph:
                # 期間中の単価推移
                place_trend(df)

                st.markdown('---')

                # ファンド毎の単価分布
                place_boxplt(df)


if __name__ == '__main__':
    main()