import os

credential_path = "conf/local/postgresql.txt"
if not os.path.exists(credential_path):
    print(f"Credentials file {credential_path} not found. Please create the file with credentials information and try again.")
    exit()

with open(credential_path) as f:
    lines = f.readlines()
    
host, dbname, user, password, port = [lines[i].strip() for i in range(5)]

with open('conf/local/credentials.yml', 'w') as f:
    f.write("db_credentials:\n")
    f.write(f"  con: postgresql://{user}:{password}@{host}:{port}/{dbname}")

print("PostgreSQL configuration file is created successfully.")