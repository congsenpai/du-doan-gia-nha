import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the trained Decision Tree model
with open('logistic.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title('House Price Prediction')

selected_MSSubClass = st.selectbox("MSSubClass:", [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190])
selected_MSZoning = st.selectbox("MSZoning:", ["A", "C", "FV", "I", "RH", "RL", "RP", "RM"])
selected_LotFrontage = st.number_input("LotFrontage:", min_value=0, max_value=1000, step=1)
selected_LotArea = st.number_input("LotArea:", min_value=0, max_value=100000, step=1)
selected_Street = st.selectbox("Street:", ["Grvl", "Pave"])
selected_Alley = st.selectbox("Alley:", ["NA", "Grvl", "Pave"])
selected_LotShape = st.selectbox("LotShape:", ["Reg", "IR1", "IR2", "IR3"])
selected_LandContour = st.selectbox("LandContour:", ["Lvl", "Bnk", "HLS", "Low"])
selected_Utilities = st.selectbox("Utilities:", ["AllPub", "NoSewr", "NoSeWa", "ELO"])
selected_LotConfig = st.selectbox("LotConfig:", ["Inside", "Corner", "CulDSac", "FR2", "FR3"])
selected_LandSlope = st.selectbox("LandSlope:", ["Gtl", "Mod", "Sev"])
selected_Neighborhood = st.selectbox("Neighborhood:", ["Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel", "Names", "NoRidge", "NPkVill", "NridgHt", "NWAmes", "OldTown", "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker"])
selected_Condition1 = st.selectbox("Condition1:", ["Artery", "Feedr", "Norm", "RRNn", "RRAn", "PosN", "PosA", "RRNe", "RRAe"])
selected_Condition2 = st.selectbox("Condition2:", ["Artery", "Feedr", "Norm", "RRNn", "RRAn", "PosN", "PosA", "RRNe", "RRAe"])
selected_BldgType = st.selectbox("BldgType:", ["1Fam", "2FmCon", "Duplx", "TwnhsE", "TwnhsI"])
selected_HouseStyle = st.selectbox("HouseStyle:", ["1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"])
selected_OverallQual = st.number_input("OverallQual:", min_value=1, max_value=10, step=1)
selected_OverallCond = st.number_input("OverallCond:", min_value=1, max_value=10, step=1)
selected_YearBuilt = st.number_input("YearBuilt:", min_value=1800, max_value=2025, step=1)
selected_YearRemodAdd = st.number_input("YearRemodAdd:", min_value=1800, max_value=2025, step=1)
selected_RoofStyle = st.selectbox("RoofStyle:", ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"])
selected_RoofMatl = st.selectbox("RoofMatl:", ["ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"])
selected_Exterior1st = st.selectbox("Exterior1st:", ["AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard", "ImStucc", "MetalSd", "Other", "Plywood", "PreCast", "Stone", "Stucco", "VinylSd", "Wd Sdng", "WdShing"])
selected_Exterior2nd = st.selectbox("Exterior2nd:", ["AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard", "ImStucc", "MetalSd", "Other", "Plywood", "PreCast", "Stone", "Stucco", "VinylSd", "Wd Sdng", "WdShing"])
selected_MasVnrType = st.selectbox("MasVnrType:", ["BrkCmn", "BrkFace", "CBlock", "None", "Stone"])
selected_MasVnrArea = st.number_input("MasVnrArea:", min_value=0, max_value=1000, step=1)
selected_ExterQual = st.selectbox("ExterQual:", ["Ex", "Gd", "TA", "Fa", "Po"])
selected_ExterCond = st.selectbox("ExterCond:", ["Ex", "Gd", "TA", "Fa", "Po"])
selected_Foundation = st.selectbox("Foundation:", ["BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"])
selected_BsmtQual = st.selectbox("BsmtQual:", ["Ex", "Gd", "TA", "Fa", "Po", "NA"])
selected_BsmtCond = st.selectbox("BsmtCond:", ["Ex", "Gd", "TA", "Fa", "Po", "NA"])
selected_BsmtExposure = st.selectbox("BsmtExposure:", ["Gd", "Av", "Mn", "No", "NA"])
selected_BsmtFinType1 = st.selectbox("BsmtFinType1:", ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"])
selected_BsmtFinSF1 = st.number_input("BsmtFinSF1:", min_value=0, max_value=5000, step=1)
selected_BsmtFinType2 = st.selectbox("BsmtFinType2:", ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"])
selected_BsmtFinSF2 = st.number_input("BsmtFinSF2:", min_value=0, max_value=5000, step=1)
selected_BsmtUnfSF = st.number_input("BsmtUnfSF:", min_value=0, max_value=5000, step=1)
selected_TotalBsmtSF = st.number_input("TotalBsmtSF:", min_value=0, max_value=10000, step=1)
selected_Heating = st.selectbox("Heating:", ["Floor", "GasA", "GasW", "Grav", "OthW", "Wall"])
selected_HeatingQC = st.selectbox("HeatingQC:", ["Ex", "Gd", "TA", "Fa", "Po"])
selected_CentralAir = st.selectbox("CentralAir:", ["N", "Y"])
selected_Electrical = st.selectbox("Electrical:", ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix"])
selected_1stFlrSF = st.number_input("1stFlrSF:", min_value=0, max_value=5000, step=1)
selected_2ndFlrSF = st.number_input("2ndFlrSF:", min_value=0, max_value=5000, step=1)
selected_LowQualFinSF = st.number_input("LowQualFinSF:", min_value=0, max_value=1000, step=1)
selected_GrLivArea = st.number_input("GrLivArea:", min_value=0, max_value=10000, step=1)
selected_BsmtFullBath = st.number_input("BsmtFullBath:", min_value=0, max_value=5, step=1)
selected_BsmtHalfBath = st.number_input("BsmtHalfBath:", min_value=0, max_value=5, step=1)
selected_FullBath = st.number_input("FullBath:", min_value=0, max_value=5, step=1)
selected_HalfBath = st.number_input("HalfBath:", min_value=0, max_value=5, step=1)
selected_BedroomAbvGr = st.number_input("BedroomAbvGr:", min_value=0, max_value=10, step=1)
selected_KitchenAbvGr = st.number_input("KitchenAbvGr:", min_value=0, max_value=10, step=1)
selected_KitchenQual = st.selectbox("KitchenQual:", ["Ex", "Gd", "TA", "Fa", "Po"])
selected_TotRmsAbvGrd = st.number_input("TotRmsAbvGrd:", min_value=0, max_value=20, step=1)
selected_Functional = st.selectbox("Functional:", ["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"])
selected_Fireplaces = st.number_input("Fireplaces:", min_value=0, max_value=5, step=1)
selected_FireplaceQu = st.selectbox("FireplaceQu:", ["Ex", "Gd", "TA", "Fa", "Po", "NA"])
selected_GarageType = st.selectbox("GarageType:", ["2Types", "Attchd", "Basment", "BuiltIn", "CarPort", "Detchd", "NA"])
selected_GarageYrBlt = st.number_input("GarageYrBlt:", min_value=1800, max_value=2025, step=1)
selected_GarageFinish = st.selectbox("GarageFinish:", ["Fin", "RFn", "Unf", "NA"])
selected_GarageCars = st.number_input("GarageCars:", min_value=0, max_value=10, step=1)
selected_GarageArea = st.number_input("GarageArea:", min_value=0, max_value=10000, step=1)
selected_GarageQual = st.selectbox("GarageQual:", ["Ex", "Gd", "TA", "Fa", "Po", "NA"])
selected_GarageCond = st.selectbox("GarageCond:", ["Ex", "Gd", "TA", "Fa", "Po", "NA"])
selected_PavedDrive = st.selectbox("PavedDrive:", ["Y", "P", "N"])
selected_WoodDeckSF = st.number_input("WoodDeckSF:", min_value=0, max_value=1000, step=1)
selected_OpenPorchSF = st.number_input("OpenPorchSF:", min_value=0, max_value=1000, step=1)
selected_EnclosedPorch = st.number_input("EnclosedPorch:", min_value=0, max_value=1000, step=1)
selected_3SsnPorch = st.number_input("3SsnPorch:", min_value=0, max_value=1000, step=1)
selected_ScreenPorch = st.number_input("ScreenPorch:", min_value=0, max_value=1000, step=1)
selected_PoolArea = st.number_input("PoolArea:", min_value=0, max_value=1000, step=1)
selected_PoolQC = st.selectbox("PoolQC:", ["Ex", "Gd", "TA", "Fa", "NA"])
selected_Fence = st.selectbox("Fence:", ["GdPrv", "MnPrv", "GdWo", "MnWw", "NA"])
selected_MiscFeature = st.selectbox("MiscFeature:", ["Elev", "Gar2", "Othr", "Shed", "TenC", "NA"])
selected_MiscVal = st.number_input("MiscVal:", min_value=0, max_value=100000, step=1)
selected_MoSold = st.number_input("MoSold:", min_value=1, max_value=12, step=1)
selected_YrSold = st.number_input("YrSold:", min_value=1800, max_value=2025, step=1)
selected_SaleType = st.selectbox("SaleType:", ["WD", "CWD", "VWD", "New", "COD", "Con", "ConLw", "ConLI", "ConLD", "Oth"])
selected_SaleCondition = st.selectbox("SaleCondition:", ["Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"])

# Tạo DataFrame mới từ các biến đã tạo
data = {
    "MSSubClass": [selected_MSSubClass],
    "MSZoning": [selected_MSZoning],
    "LotFrontage": [selected_LotFrontage],
    "LotArea": [selected_LotArea],
    "Street": [selected_Street],
    "Alley": [selected_Alley],
    "LotShape": [selected_LotShape],
    "LandContour": [selected_LandContour],
    "Utilities": [selected_Utilities],
    "LotConfig": [selected_LotConfig],
    "LandSlope": [selected_LandSlope],
    "Neighborhood": [selected_Neighborhood],
    "Condition1": [selected_Condition1],
    "Condition2": [selected_Condition2],
    "BldgType": [selected_BldgType],
    "HouseStyle": [selected_HouseStyle],
    "OverallQual": [selected_OverallQual],
    "OverallCond": [selected_OverallCond],
    "YearBuilt": [selected_YearBuilt],
    "YearRemodAdd": [selected_YearRemodAdd],
    "RoofStyle": [selected_RoofStyle],
    "RoofMatl": [selected_RoofMatl],
    "Exterior1st": [selected_Exterior1st],
    "Exterior2nd": [selected_Exterior2nd],
    "MasVnrType": [selected_MasVnrType],
    "MasVnrArea": [selected_MasVnrArea],
    "ExterQual": [selected_ExterQual],
    "ExterCond": [selected_ExterCond],
    "Foundation": [selected_Foundation],
    "BsmtQual": [selected_BsmtQual],
    "BsmtCond": [selected_BsmtCond],
    "BsmtExposure": [selected_BsmtExposure],
    "BsmtFinType1": [selected_BsmtFinType1],
    "BsmtFinSF1": [selected_BsmtFinSF1],
    "BsmtFinType2": [selected_BsmtFinType2],
    "BsmtFinSF2": [selected_BsmtFinSF2],
    "BsmtUnfSF": [selected_BsmtUnfSF],
    "TotalBsmtSF": [selected_TotalBsmtSF],
    "Heating": [selected_Heating],
    "HeatingQC": [selected_HeatingQC],
    "CentralAir": [selected_CentralAir],
    "Electrical": [selected_Electrical],
    "1stFlrSF": [selected_1stFlrSF],
    "2ndFlrSF": [selected_2ndFlrSF],
    "LowQualFinSF": [selected_LowQualFinSF],
    "GrLivArea": [selected_GrLivArea],
    "BsmtFullBath": [selected_BsmtFullBath],
    "BsmtHalfBath": [selected_BsmtHalfBath],
    "FullBath": [selected_FullBath],
    "HalfBath": [selected_HalfBath],
    "BedroomAbvGr": [selected_BedroomAbvGr],
    "KitchenAbvGr": [selected_KitchenAbvGr],
    "KitchenQual": [selected_KitchenQual],
    "TotRmsAbvGrd": [selected_TotRmsAbvGrd],
    "Functional": [selected_Functional],
    "Fireplaces": [selected_Fireplaces],
    "FireplaceQu": [selected_FireplaceQu],
    "GarageType": [selected_GarageType],
    "GarageYrBlt": [selected_GarageYrBlt],
    "GarageFinish": [selected_GarageFinish],
    "GarageCars": [selected_GarageCars],
    "GarageArea": [selected_GarageArea],
    "GarageQual": [selected_GarageQual],
    "GarageCond": [selected_GarageCond],
    "PavedDrive": [selected_PavedDrive],
    "WoodDeckSF": [selected_WoodDeckSF],
    "OpenPorchSF": [selected_OpenPorchSF],
    "EnclosedPorch": [selected_EnclosedPorch],
    "3SsnPorch": [selected_3SsnPorch],
    "ScreenPorch": [selected_ScreenPorch],
    "PoolArea": [selected_PoolArea],
    "PoolQC": [selected_PoolQC],
    "Fence": [selected_Fence],
    "MiscFeature": [selected_MiscFeature],
    "MiscVal": [selected_MiscVal],
    "MoSold": [selected_MoSold],
    "YrSold": [selected_YrSold],
    "SaleType": [selected_SaleType],
    "SaleCondition": [selected_SaleCondition]
}

df = pd.DataFrame(data)
#  viết hàm biến đổi tất các các cột kiểu object thành kiểu int
def convert_object_columns_to_int(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]
    return df
def encoded_input_data(df):
    df['MSSubClass']=df['MSSubClass'].astype('object')
    df['YrSold']=df['YrSold'].astype('object')
    df['LotFrontage']=df['LotFrontage'].astype('float')
    convert_object_columns_to_int(df)
    return df
encoded_input_data(df)

# 2. Sử dụng mô hình để dự đoán
prediction = model.predict(df)

# 3. Xuất dự đoán ra giao diện người dùng
st.write("Dự đoán giá nhà của bạn là:", prediction[0])

# 4. Xuất dự đoán ra một tệp CSV
output_df = pd.DataFrame({'Prediction': prediction})
output_df.to_csv('house_price_prediction.csv', index=False)

