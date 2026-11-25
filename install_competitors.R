if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes", repos="https://cloud.r-project.org")
repos <- c(
  "gaofengnan/charcoal@713a87c2fbb943cedd0e0df9a2c37c9d78689a6a",
  "HaotianXu/changepoints@002b97ea4e29b38dbd55ccfcbbf24ffeccfd9532",
  "tobiaskley/inferchange@07f8e3fbb20bb3275c777a9e1bfb1cb37a48e656",
  "Dom-Owens-UoB/moseg@ea02a939add8425ae0fd8d387e7b5d2094ad4587"
)
for (repo in repos) remotes::install_github(repo, upgrade="never")
remotes::install_local("myglm-3", upgrade="never")
