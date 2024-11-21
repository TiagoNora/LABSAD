@echo off
set MONGO_URI=mongodb://mongoadmin:a79c987b4dce244e9bc21620@vsgate-s1.dei.isep.ipp.pt:10777
set DB_NAME=labsad
set COLLECTION_NAME=stockPrices
set CSV_DIRECTORY=C:\Users\João\Desktop\Projetos\LABSAD\files  # Update with your actual CSV folder path

for %%f in (%CSV_DIRECTORY%\*.csv) do (
    echo Importing %%f into MongoDB...
    mongoimport --uri=%MONGO_URI% --db=%DB_NAME% --collection=%COLLECTION_NAME% --type=csv --file="%%f" --headerline
    echo %%f imported successfully.
)

echo All CSV files have been imported.
