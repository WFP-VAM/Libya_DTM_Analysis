import pandas as pd
import numpy as np

data = pd.read_excel("Data/Raw/DTM_FMS_F2_FS_dataset_20190312_fin_dateformat.xlsx",
                     sheet_name="DataSet_FMS2_FS", parse_dates=True)

recode_dic = {
    "Did not answer": np.NaN,
    "Don’t want to answer": np.NaN,
    0.55: np.NaN,  # Not applicable
    0.88: np.NaN,  # Did not answer
    0.22: np.NaN,  # No selection made
    'No': 0,  # Coping questions
    'Yes': 1  # Coping questions
}

data = data.replace(recode_dic)


def cat_to_dummies(df, col, suff=False):
    smalldf = pd.get_dummies(df[col], dummy_na=True)
    # Hack to keep missing values
    smalldf.loc[smalldf[np.NaN] == 1, :] = np.NaN
    smalldf = smalldf.drop(np.NaN, axis=1)
    if suff:
        smalldf.columns = smalldf.columns + suff
    df = pd.concat([df, smalldf], axis=1)
    return df


# Time in Libya
data['10_arrival_date_libya'] = pd.to_datetime(
    data['10_arrival_date_libya'], errors='coerce')
data['date'] = pd.to_datetime(data['date'], errors='coerce')

data['time_in_libya'] = data['date'] - data['10_arrival_date_libya']
data.loc[data['time_in_libya'].dt.days <= 0, 'time_in_libya'] = np.NaN

data['months_in_libya'] = data['time_in_libya'].dt.days / 30.44

data['time_libya_cat'] = np.NaN
data.loc[(data['time_in_libya'].dt.days > 0) & (
    data['time_in_libya'].dt.days < 183), 'time_libya_cat'] = 'less than 6m'
data.loc[(data['time_in_libya'].dt.days >= 183) & (
    data['time_in_libya'].dt.days < 365), 'time_libya_cat'] = 'between 6m and 1y'
data.loc[(data['time_in_libya'].dt.days >= 365) & (
    data['time_in_libya'].dt.days < 730), 'time_libya_cat'] = 'between 1y and 2y'
data.loc[(data['time_in_libya'].dt.days >= 730) & (
    data['time_in_libya'].dt.days < 10000), 'time_libya_cat'] = 'more than 2y'

# Age categories
data['age_cat'] = np.NaN
data.loc[(data['4_age'] > 0) & (
    data['4_age'] < 20), 'age_cat'] = '<20 years'
data.loc[(data['4_age'] >= 20) & (
    data['4_age'] < 30), 'age_cat'] = '20-30 years'
data.loc[(data['4_age'] >= 30) & (
    data['4_age'] < 40), 'age_cat'] = '30-40 years'
data.loc[(data['4_age'] >= 40) & (
    data['4_age'] < 100), 'age_cat'] = '>40 years'

# Food Consumption Analysis
dic_var = {
    'i_1.1_food_consum_cereals': "cereals",
    'i_1.2_food_consum_legumes': "legumes",
    'i_1.3_food_consum_vegetables': "veggies",
    'i_1.4_food_consum_fruits': "fruits",
    'i_1.5_food_consum_meat': "meat",
    'i_1.6_food_consum_milk': "dairy",
    'i_1.7_food_consum_oil': "fats",
    'i_1.8_food_consum_sugar': "sugar"}

food_items = ['cereals',
              'legumes',
              'veggies',
              'fruits',
              'meat',
              'dairy',
              'fats',
              'sugar']

data = data.rename(dic_var, axis=1)


def FCS(df):
    """
    Compute the FCS with standard weights and variable names.
    Args:
        df: Pandas DataFrame
    Returns:
        A Pandas Series
    """
    FCS = 2 * df["cereals"] + 3 * df["legumes"] + 1 * df["veggies"] + 1 * df["fruits"] + \
        4 * df["meat"] + 4 * df["dairy"] + 0.5 * df["sugar"] + 0.5 * df["fats"]
    return FCS


def fcg_groups(col, a, b):
    """
    Create a Categorical variable from a continuous variable based on 2 thresholds
    Args:
        col: A Pandas Series
        a: lower threshold
        b: upper threshold
    Returns:
        A Pandas Series
    """
    if col <= a:
        return "Poor"
    if (col > a) & (col <= b):
        return "Borderline"
    if col > b:
        return "Acceptable"
    return np.NaN


data['FCS'] = data.apply(FCS, axis=1)
data["FCG"] = data["FCS"].apply(fcg_groups, args=(28, 42))

data = cat_to_dummies(data, 'FCG')

# Remittances Analysis
# How much have you sent home in dollar since left?

# Do they want to answer?
data['remit_answered'] = 0
data.loc[data['27_remit_amount_sent'].notnull(), 'remit_answered'] = 1

# Yes/No did they send money?
data['money_sent_dum'] = np.NaN
data.loc[(data['27_remit_amount_sent'] == 0), 'money_sent_dum'] = 0
data.loc[(data['27_remit_amount_sent'] > 0) & (
    data['27_remit_amount_sent'] < 100000), 'money_sent_dum'] = 1

# Let's consider the 0 has missing to get the median expenditures
data['remit_amount_pos'] = np.NaN
data.loc[(data['27_remit_amount_sent'] == 0), 'remit_amount_pos'] = np.NaN
data.loc[(data['27_remit_amount_sent'] > 0) & (data['27_remit_amount_sent']
                                               < 100000), 'remit_amount_pos'] = data['27_remit_amount_sent']

# Amount sent per month
data['remit_amount_pos_permonth'] = data['remit_amount_pos'] / \
    data['months_in_libya']

# Reasons sending money
reason_remit = data.columns[data.columns.str.startswith('28')].tolist()
means_remit = data.columns[(data.columns.str.startswith('29')) & (
    data.columns != '29.0.6_remit_means_text')]

# Coping
data = cat_to_dummies(data, 'i_2.1_compromise_food_why')

coping_var = ["i_2_compromise_food_consumpt",
              "To save money",
              "To send money back home",
              "For accommodation",
              "For health reasons",  # Only considering top responses for categorical variables
              "i_3.1_coping_exchange",
              "i_3.2_coping_scavenge",
              "i_3.3_coping_borrow",
              "i_3.4_coping_illegal",
              "i_3.5_coping_begging",
              "i_3.6_coping_child_labor",
              "i_3.7_coping_sold",
              "i_3.8_coping_spent_savings",
              "i_3.9_coping_work_food"]

# Livelihood
data = cat_to_dummies(data, '7_employ_status_before', suff='_coo')
data = cat_to_dummies(data, '7.1.1_occupation_coo', suff='_coo')
data = cat_to_dummies(data, '8_employment_status_libya', suff='_lib')
data = cat_to_dummies(data, '8.1.1_occupation_libya', suff='_lib')

data['same_job_libya_coo'] = 0
data.loc[data['7.1.1_occupation_coo']
         == data['8.1.1_occupation_libya'], 'same_job_libya_coo'] = 1
data.loc[(data['7.1.1_occupation_coo'] == np.NaN) | (
    data['8.1.1_occupation_libya'] == np.NaN), 'same_job_libya_coo'] = np.NaN

liv = ["Unemployed and looking for job_coo",  # Only considering top responses for categorical variables
       "Employed_coo",
       "Self-Employed_coo",
       "Construction, Water Supply, Electricity, Gas_coo",
       "Agriculture, Pastoralism, Fishing, Food Industry_coo",
       "Craft_coo",
       "Other_coo",
       "Retail, Sales_coo",
       "Plant and machine operators, and assemblers, mechanicals_coo",
       "Domestic work_coo",
       "7.2_droughts_floods",
       "same_job_libya_coo",
       "Unemployed and looking for job_lib",
       "Employed_lib",
       "Self-Employed_lib",
       "Construction, Water Supply, Electricity, Gas_lib",
       "Agriculture, Pastoralism, Fishing, Food Industry_lib",
       "Craft_lib",
       "Other_lib",
       "Retail, Sales_lib",
       "Plant and machine operators, and assemblers, mechanicals_lib",
       "Domestic work_lib"]

# Analysis
data['no_group'] = 'All data'

grp_dic = {
    'months_in_libya': ["count", "mean", "median"],
    'remit_answered': "mean",
    'money_sent_dum': "mean",
    # Consider the median to deal with outlier
    'remit_amount_pos': ["count", "median"],
    'remit_amount_pos_permonth': "median"  # Same
}
grp_dic.update(dict.fromkeys(reason_remit, "mean"))
grp_dic.update(dict.fromkeys(means_remit, "mean"))
grp_dic.update({
    'FCS': ["count", "mean"],
    'Poor': "mean",
    'Borderline': "mean",
    'Acceptable': "mean"
})

grp_dic.update(dict.fromkeys(food_items, "mean"))
grp_dic.update(dict.fromkeys(coping_var, ["count", "mean"]))

grp_dic.update(dict.fromkeys(liv, ["count", "mean"]))

cross_vars = ['no_group', 'p1_geodivision', ['p1_geodivision', 'p2_mantika'], '2.0_region_of_origin', [
    '2.0_region_of_origin', '2.1_nationality'], '3_sex', 'age_cat', '5_marital_status', '6_edu_level']

# Exporting results to Excel
with pd.ExcelWriter("Data/Aggregated/coping_livelihood_remittances_fcs_analysis.xlsx") as writer:
    for x in cross_vars:
        group = data.groupby(x)
        summary_stats = group.agg(grp_dic)
        summary_stats['n'] = group.size()
        summary_stats = summary_stats[summary_stats['n'] > 30]
        if isinstance(x, list):
            summary_stats.to_excel(writer, sheet_name=x[1])
        else:
            summary_stats.to_excel(writer, sheet_name=x)
