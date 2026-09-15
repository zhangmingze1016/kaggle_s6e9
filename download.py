import kagglehub

path = kagglehub.competition_download(
    "playground-series-s6e9"
)

print("Path to competition files:", path)