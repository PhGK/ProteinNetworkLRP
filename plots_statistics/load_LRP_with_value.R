library(ggplot2)
library(stringr)
library(magrittr)
library(tidyr)
library(dplyr)
library(parallel)
library(DescTools)
library(gplots)
library(pbmcapply)
library(data.table)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

numCores <- detectCores()
setwd('.') # wd to relprop result files
RAWPATH = '../results/LRP/raw_data/'
USEPATH = '../results/LRP/use_data/'
dir.create(USEPATH)

filenames <- list.files(RAWPATH)
all_data <-rbindlist(lapply(filenames, function(filename) fread(paste0(RAWPATH, filename)))) %>%
  dplyr::select(-V1)
                  
ORGANS <- read.csv('../data/tcpa_data_051017.csv', check.names = F) %>%
  dplyr::select(ID, Cancer_Type)
all_data_ORGAN <- inner_join(ORGANS, all_data, by = c('ID'='sample_name') ) 

print(length(filenames))

#write.csv(all_data_ORGAN,paste0(USEPATH, 'all_data.csv'), row.names=F)
print(dim(all_data))


all_data_ORGAN %>% dplyr::arrange(desc(LRP)) %>% .[5,]
all_data_ORGAN$LRP %>% max

all_data_ORGAN$LRP %>% min
all_data_ORGAN %>% dplyr::arrange(LRP) %>% .[5,]

all_data_ORGAN$LRP %>% IQR
