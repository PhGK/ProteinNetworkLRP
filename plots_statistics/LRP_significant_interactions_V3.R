library(ggplot2)
library(stringr)
library(magrittr)
library(tidyr)
library(dplyr)
library(DescTools)
library(gplots)
library(ComplexHeatmap)
library(pbmcapply)
library(circlize)
library(data.table)
library(Hmisc)
library(stringi)
library(mvMORPH)
library(stats)
library(xtable)
library(plyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

test_data <- fread('../results/LRP/use_data/all_data.csv')
test_data$LRP <- test_data$LRP %>% abs()
##################
##############################################################################################################################
#load ground truth data
protein_data <- read.csv('../data/tcpa_data_051017.csv', check.names = F) %>% 
  pivot_longer(!c(ID,Cancer_Type), names_to = "proteins", values_to = "expression") %>% 
  dplyr::select(-Cancer_Type)


for_correlation <- test_data %>% dplyr::select(ID, y, y_pred) %>%
  unique() %>%
  group_by(ID)

correlation <- ddply(for_correlation, "ID", summarize, "corr" = cor(y, y_pred))
highcorrelation <- correlation %>% filter(corr>0.5)

rcorr(for_correlation$y, for_correlation$y_pred)

case_numbers_old <- test_data %>% group_by(ID) %>% dplyr::summarize("Cancer_Type" = max(Cancer_Type))
summary(as.factor(case_numbers_old$Cancer_Type))


test_data <- test_data %>% filter(ID %in% highcorrelation$ID)

########################################
#facet wrap of highest values
#SYMMETRIZED

# not grouped by ORGANS!!
number_interactions = 36
################

##################
#symmetrize data
LRP_dir <- test_data %>% dplyr::select(ID, Cancer_Type, predicting_protein, masked_protein, LRP)
LRP_trans <- LRP_dir
colnames(LRP_trans) <- c('ID','Cancer_Type', "masked_protein", "predicting_protein", "tLRP")

sym_LRP <- left_join(LRP_dir, LRP_trans) %>% mutate("LRP_sym" = 0.5*(LRP+tLRP)) %>% 
  arrange(desc(LRP_sym)) %>% filter(predicting_protein >=masked_protein) %>% dplyr::select(-c(LRP, tLRP))

sym_highest_LRP <- sym_LRP %>% 
  group_by(predicting_protein, masked_protein) %>%
  dplyr::summarize("meanLRP" = median(LRP_sym)) %>% 
  ungroup() %>%
  arrange(desc(meanLRP)) %>% dplyr::select(predicting_protein, masked_protein, meanLRP) 
############################
############################
#compare with reactome
setwd('../data/')
adj_react <- read.delim('int_react_147_060418.csv', header = F)  
colnames(adj_react) <- c("masked_protein", as.character(adj_react$V1))
adj_react_matrix <- as.matrix(adj_react[,-1])
rownames(adj_react_matrix) <- as.character(adj_react$masked_protein)

# order names alphanumerically
adj_react_matrix_ordered <- adj_react_matrix[str_order(rownames(adj_react_matrix)), str_order(colnames(adj_react_matrix))]
adj_react_long <- pivot_longer(adj_react,cols=!masked_protein, names_to = "predicting_protein", values_to = "edge")
adj_react_long$predicting_protein <- as.character(adj_react_long$predicting_protein)
adj_react_long$masked_protein <- as.character(adj_react_long$masked_protein)

adj_react_long_sym <- adj_react_long %>% filter((predicting_protein) > (masked_protein))
k = 100
full_frame <- left_join(adj_react_long_sym, sym_highest_LRP, by =c("masked_protein", "predicting_protein")) %>% 
  dplyr::arrange(desc(meanLRP)) %>% .[1:k,]

#####
#stat hypergeometric test
library(stats)
ncorrect <- full_frame$edge %>% sum()
ncorrectall <- adj_react_long_sym$edge %>% sum()
nfalseall <- (1-adj_react_long_sym$edge) %>% sum()

phyper(ncorrect-1, ncorrectall, nfalseall, k, lower.tail = F, log.p = FALSE)
ncorrect
#############################

# filter out phosphorylation variants
highest_names <- tidyr::extract(sym_highest_LRP, predicting_protein, into ="p", "([^_]*).*", remove = F) %>%
  tidyr::extract(masked_protein, into ="m", "([^_]*).*", remove = F) %>%
  dplyr::filter(p!=m) %>%
  dplyr::select(-c(p,m)) %>%
  tidyr::extract(predicting_protein, into ="p", "(..).*", remove = F) %>%
  tidyr::extract(masked_protein, into ="m", "(..).*", remove = F) %>%
  dplyr::filter(p!=m) %>%
  dplyr::select(-c(p,m)) %>%
  ungroup %>%
  dplyr::select(predicting_protein, masked_protein, meanLRP)
############################
#compare with reactome

full_frame <- left_join(adj_react_long_sym, highest_names, by =c("masked_protein", "predicting_protein")) %>% 
  dplyr::arrange(desc(meanLRP)) %>% .[1:36,]

#####
#stat hypergeometric test
library(stats)
ncorrect <- full_frame$edge %>% sum()
ncorrectall <- adj_react_long_sym$edge %>% sum()
nfalseall <- (1-adj_react_long_sym$edge) %>% sum()
k = 36
phyper(ncorrect-1, ncorrectall, nfalseall, 36, lower.tail = F, log.p = FALSE)
ncorrect
#############################

h1 <- highest_names[1:36,] %>% left_join(adj_react_long_sym, by = c("masked_protein", "predicting_protein")) %>%
  dplyr::mutate("predicting_protein" = ifelse(edge==1, paste(predicting_protein, '*', sep=''), predicting_protein))

sym_LRP <- sym_LRP %>% left_join(adj_react_long_sym, by = c("masked_protein", "predicting_protein")) %>%
  dplyr::mutate("predicting_protein" = ifelse(edge==1, paste(predicting_protein, '*', sep=''), predicting_protein))
subset_data <- left_join(h1, sym_LRP) %>% dplyr::group_by(predicting_protein, masked_protein, Cancer_Type) %>%
  dplyr::summarize("meanLRP" = median(LRP_sym))
########################

medians_IQR <- left_join(h1, sym_LRP) %>%dplyr::select(-Cancer_Type) %>% dplyr::group_by(predicting_protein, masked_protein) %>%
  dplyr::summarize("meanLRP" = median(LRP_sym), "IQR" = IQR(LRP_sym))

anova_data <- left_join(h1, sym_LRP)

myanova <- function(id) {
  curr_h <- h1[id,]
  subset <- anova_data %>% filter(predicting_protein == curr_h$predicting_protein, masked_protein == curr_h$masked_protein)
  kruskal_result <- kruskal.test(LRP_sym ~ Cancer_Type, data = subset)
  c(curr_h$predicting_protein, curr_h$masked_protein, kruskal_result$p.value)
  #kruskal_result
}

anovavalues <- sapply(seq(36), myanova)
adj <- p.adjust(anovavalues[3,]) %>% format(digits=2)
adj_values <- rbind(anovavalues, adj) %>% t() %>% data.frame()
colnames(adj_values) <- c("predicting_protein", "masked_protein", "pvalue", "adjpvalue")
description <- left_join(medians_IQR, adj_values)
description$Cancer_Type = "ACC"
################################
high_names <- subset_data %>% ungroup %>% group_by(predicting_protein, masked_protein) %>%
  filter(meanLRP >= 0.8*max(meanLRP)) %>% ungroup()

plotobject <- ggplot(subset_data, aes(x = Cancer_Type, y =meanLRP, fill = Cancer_Type), color="black") +
  #geom_line(aes(group=1)) 
  geom_bar(stat="identity", color="black") #geom_boxplot(outlier.shape = NA) 

plotobject2 <- plotobject + 
  geom_text(data = high_names, aes(label = Cancer_Type),hjust = -0.3, angle = 90) + 
  facet_wrap( ~ masked_protein+predicting_protein, nrow = 6) +
  theme_bw()+
  theme(axis.text.x = element_blank(), 
        strip.background = element_blank(),
        strip.text  = element_text(size=11),
        axis.title = element_text(size=15),
        legend.text = element_text(size=15),
        legend.title = element_text(size=18))+
  scale_y_continuous(expand= expansion(c(0,0.6)))+
  ylab("median LRP")+
  xlab("Cancer")+
  labs(fill = "Cancer")

plotobject2 <- plotobject + 
  geom_text(data = high_names, aes(label = Cancer_Type),hjust = -0.3, angle = 90) + 
  facet_wrap( ~ masked_protein+predicting_protein, nrow = 6) +
  geom_text(data = description, aes(x = 10, y = 0.18, label = paste('median: ', round(meanLRP, digits=3), 'IQR: ', round(IQR, digits=3)))) + 
  geom_text(data = description, aes(x = 11, y = 0.16, label = paste('p: ', adjpvalue))) + 
  theme_bw()+
  theme(axis.text.x = element_blank(), 
        strip.background = element_blank(),
        strip.text  = element_text(size=11),
        axis.title = element_text(size=15),
        legend.text = element_text(size=15),
        legend.title = element_text(size=18))+
  scale_y_continuous(expand= expansion(c(0,0.1)))+
  ylab("median LRP")+
  xlab("Cancer")+
  labs(fill = "Cancer")

png("../plots_statistics/figures/highest_wrap.png",width = 1000, height = 1000)
plotobject2
dev.off()
plotobject2

#########################################################################################
h2 <- highest_names[37:72,]

subset <- left_join(h2, sym_LRP) %>% dplyr::group_by(predicting_protein, masked_protein, ORGAN) %>%
  dplyr::summarize("meanLRP" = median(LRP_sym))

######

medians_IQR <- left_join(h2, sym_LRP) %>%dplyr::select(-ORGAN) %>% dplyr::group_by(predicting_protein, masked_protein) %>%
  dplyr::summarize("meanLRP" = median(LRP_sym), "IQR" = IQR(LRP_sym))

anova_data <- left_join(h2, sym_LRP)

myanova <- function(id) {
  curr_h <- h2[id,]
  subset <- anova_data %>% filter(predicting_protein == curr_h$predicting_protein, masked_protein == curr_h$masked_protein)
  kruskal_result <- kruskal.test(LRP_sym ~ ORGAN, data = subset)
  c(curr_h$predicting_protein, curr_h$masked_protein, kruskal_result$p.value)
  #kruskal_result
}

anovavalues <- sapply(seq(36), myanova)
adj <- p.adjust(anovavalues[3,]) %>% format(digits=2)
adj_values <- rbind(anovavalues, adj) %>% t() %>% data.frame()
colnames(adj_values) <- c("predicting_protein", "masked_protein", "pvalue", "adjpvalue")
description <- left_join(medians_IQR, adj_values)
description$ORGAN = "ACC"



#######
high_names <- subset %>% ungroup %>% group_by(predicting_protein, masked_protein) %>%
  filter(meanLRP >= 0.8*max(meanLRP)) %>% ungroup()

plotobject <- ggplot(subset, aes(x = ORGAN, y =meanLRP, fill = ORGAN), color="black") +
  #geom_line(aes(group=1)) 
  geom_bar(stat="identity", color="black") #geom_boxplot(outlier.shape = NA) 

plotobject2 <- plotobject + 
  geom_text(data = high_names, aes(label = ORGAN),hjust = -0.3, angle = 90) + 
  facet_wrap( ~ masked_protein+predicting_protein, nrow = 6) +
  theme_bw()+
  theme(axis.text.x = element_blank(), 
        strip.background = element_blank(),
        strip.text  = element_text(size=11),
        axis.title = element_text(size=15),
        legend.text = element_text(size=15),
        legend.title = element_text(size=18))+
  scale_y_continuous(expand= expansion(c(0,0.6)))+
  ylab("median LRP")+
  xlab("Cancer")+
  labs(fill = "Cancer")

plotobject2 <- plotobject + 
  geom_text(data = high_names, aes(label = ORGAN),hjust = -0.3, angle = 90) + 
  facet_wrap( ~ masked_protein+predicting_protein, nrow = 6) +
  geom_text(data = description, aes(x = 10, y = 18, label = paste('median: ', round(meanLRP, digits=1), 'IQR: ', round(IQR, digits=1)))) + 
  geom_text(data = description, aes(x = 10, y = 16, label = paste('p: ', adjpvalue))) + 
  theme_bw()+
  theme(axis.text.x = element_blank(), 
        strip.background = element_blank(),
        strip.text  = element_text(size=11),
        axis.title = element_text(size=15),
        legend.text = element_text(size=15),
        legend.title = element_text(size=18))+
  scale_y_continuous(expand= expansion(c(0,0.1)))+
  ylab("median LRP")+
  xlab("Cancer")+
  labs(fill = "Cancer")

png("/mnt/scratch2/mlprot/mlprot_220920/plots_statistics/figures/highest_wrap2.png",width = 1000, height = 1000)
plotobject2
dev.off()
###
h3 <- highest_names[73:108,]

subset <- left_join(h3, sym_LRP) %>% dplyr::group_by(predicting_protein, masked_protein, ORGAN) %>%
  dplyr::summarize("meanLRP" = median(LRP_sym))

######

medians_IQR <- left_join(h3, sym_LRP) %>%dplyr::select(-ORGAN) %>% dplyr::group_by(predicting_protein, masked_protein) %>%
  dplyr::summarize("meanLRP" = median(LRP_sym), "IQR" = IQR(LRP_sym))

anova_data <- left_join(h3, sym_LRP)

myanova <- function(id) {
  curr_h <- h3[id,]
  subset <- anova_data %>% filter(predicting_protein == curr_h$predicting_protein, masked_protein == curr_h$masked_protein)
  kruskal_result <- kruskal.test(LRP_sym ~ ORGAN, data = subset)
  c(curr_h$predicting_protein, curr_h$masked_protein, kruskal_result$p.value)
  #kruskal_result
}

anovavalues <- sapply(seq(36), myanova)
adj <- p.adjust(anovavalues[3,]) %>% format(digits=2)
adj_values <- rbind(anovavalues, adj) %>% t() %>% data.frame()
colnames(adj_values) <- c("predicting_protein", "masked_protein", "pvalue", "adjpvalue")
description <- left_join(medians_IQR, adj_values)
description$ORGAN = "ACC"



#######

high_names <- subset %>% ungroup %>% group_by(predicting_protein, masked_protein) %>%
  filter(meanLRP >= 0.8*max(meanLRP)) %>% ungroup()

plotobject <- ggplot(subset, aes(x = ORGAN, y =meanLRP, fill = ORGAN), color="black") +
  #geom_line(aes(group=1)) 
  geom_bar(stat="identity", color="black") #geom_boxplot(outlier.shape = NA) 

plotobject2 <- plotobject + 
  geom_text(data = high_names, aes(label = ORGAN),hjust = -0.3, angle = 90) + 
  facet_wrap( ~ masked_protein+predicting_protein, ncol = 6) +
  theme_bw()+
  theme(axis.text.x = element_blank(), 
        strip.background = element_blank(),
        strip.text  = element_text(size=11),
        axis.title = element_text(size=15),
        legend.text = element_text(size=15),
        legend.title = element_text(size=18))+
  scale_y_continuous(expand= expansion(c(0,0.6)))+
  ylab("median LRP")+
  xlab("Cancer")+
  labs(fill = "Cancer")

plotobject2 <- plotobject + 
  geom_text(data = high_names, aes(label = ORGAN),hjust = -0.3, angle = 90) + 
  facet_wrap( ~ masked_protein+predicting_protein, nrow = 6) +
  geom_text(data = description, aes(x = 10, y = 18, label = paste('median: ', round(meanLRP, digits=1), 'IQR: ', round(IQR, digits=1)))) + 
  geom_text(data = description, aes(x = 10, y = 16, label = paste('p: ', adjpvalue))) + 
  theme_bw()+
  theme(axis.text.x = element_blank(), 
        strip.background = element_blank(),
        strip.text  = element_text(size=11),
        axis.title = element_text(size=15),
        legend.text = element_text(size=15),
        legend.title = element_text(size=18))+
  scale_y_continuous(expand= expansion(c(0,0.1)))+
  ylab("median LRP")+
  xlab("Cancer")+
  labs(fill = "Cancer")

png("/mnt/scratch2/mlprot/mlprot_220920/plots_statistics/figures/highest_wrap3.png",width = 1000, height = 1000)
plotobject2
dev.off()


############################################
####################################
#test LRP values of phosphorylated variants


highest_LRP <- sym_LRP %>% 
  tidyr::extract(masked_protein, into ="m", "([^_]*).*", remove = F) %>%
  tidyr::extract(predicting_protein, into ="p", "([^_]*).*", remove = F) %>%
  filter(predicting_protein!=masked_protein) %>%
  mutate("phosphointeract" = ifelse(m==p,1,0))

result <- wilcox.test(LRP_sym ~ phosphointeract, highest_LRP)
result2 <- t.test(LRP_sym ~ phosphointeract, highest_LRP)
result
result2

phos <- highest_LRP %>% filter(m==p) %>% filter(predicting_protein != masked_protein) %>% .$LRP
nphos <- highest_LRP %>% filter(m!=p) %>% filter(predicting_protein != masked_protein) %>% .$LRP
length(nphos)

median(highest_LRP %>% filter(phosphointeract ==1) %>% .$LRP)
IQR(highest_LRP %>% filter(phosphointeract ==1) %>% .$LRP)
median(highest_LRP %>% filter(phosphointeract ==0) %>% .$LRP)
IQR(highest_LRP %>% filter(phosphointeract ==0) %>% .$LRP)

ggplot(highest_LRP, aes(x = LRP_sym, fill = as.factor(phosphointeract))) + geom_histogram(aes(y = after_stat(density), alpha = 0.1), bins = 500)
ggplot(highest_LRP %>% arrange(LRP_sym), aes(y = LRP_sym, x = seq(length(LRP_sym)))) + geom_point()


#########################################################
#test interactions between c-MET caspase8 parpcleaved Snail ercc1
library(ggExtra)
selprots <- c("CMET", "CASPASE8", "PARPCLEAVED", "SNAIL", "ERCC1")

to_sym_LRP <- test_data %>% dplyr::select(predicting_protein, masked_protein, LRP, ORGAN)
to_sym_LRP_t <- to_sym_LRP
colnames(to_sym_LRP_t) <- c("masked_protein", "predicting_protein", "tLRP", "ORGAN")
sym_highest_LRP <- left_join(to_sym_LRP, to_sym_LRP_t) %>% mutate("LRP_new" = 0.5*(LRP+tLRP)) #%>% 
  arrange(desc(LRP)) %>% filter(predicting_protein <masked_protein)


quintum <- sym_highest_LRP %>% 
  dplyr::filter(predicting_protein %in% selprots, masked_protein %in% selprots, predicting_protein!=masked_protein) %>%
  dplyr::select(case, ORGAN, predicting_protein, masked_protein, LRP) %>%
  unite("interaction",predicting_protein:masked_protein) %>%
  pivot_wider(names_from = interaction, values_from = LRP)

quintsne <- Rtsne(quintum[,-c(1,2)])
tsneplot<- data.frame(ORGAN=quintum$ORGAN, x=quintsne$Y[,1], y=quintsne$Y[,2])
ggplot(tsneplot, aes(x=x, y=y, color = ORGAN)) + geom_point()


quintum <-test_data %>% ungroup()%>%
  dplyr::filter(predicting_protein %in% selprots, masked_protein %in% selprots, predicting_protein!=masked_protein) %>%
  dplyr::select(case, ORGAN, predicting_protein, masked_protein, LRP) 

t_quintum <- quintum
colnames(t_quintum) <- c("case", "ORGAN", "masked_protein", "predicting_protein", "tLRP")

symmetric_quintum <- left_join(quintum, t_quintum) %>% mutate("new_LRP" = 0.5*(LRP+tLRP)) %>% filter(predicting_protein < masked_protein)
library('scales')
marginplots <- function(selORGAN) { 
  quintum_raw <-  symmetric_quintum %>% 
    dplyr::select(case, ORGAN, predicting_protein, masked_protein, new_LRP) %>% 
    unite("interaction",predicting_protein:masked_protein) %>% mutate(logLRP = log(1+new_LRP)) %>% filter(ORGAN==selORGAN)
  
  p <- ggplot(quintum_raw, aes(x=as.integer(as.factor(interaction)), y = logLRP)) + 
    geom_point(alpha = 0.0) + geom_line(aes(color = case), show.legend = F) +
    ggtitle(selORGAN) +
    ylim(0,6)+ 
    theme(plot.title = element_text(size=15, margin = margin(10,0, 0,0)),
          axis.title.x = element_blank(),
          plot.margin = margin(0,0,0,0))
  p2 <- ggMarginal(p, type="histogram", margins = "y")
             
  p2
}

organs <- symmetric_quintum$ORGAN %>% unique()%>% unlist()

marplots <- lapply(organs, marginplots)
library(ggpubr)
png("/mnt/scratch2/mlprot/mlprot_220920/plots_statistics/figures/quintum.png",width = 1000, height = 1000)
ggarrange(plotlist=marplots)
dev.off()

##correlation
quintum_raw <-  symmetric_quintum %>% 
  unite("interaction",predicting_protein:masked_protein) %>% 
  dplyr::select(case, interaction, new_LRP) %>%
  #mutate("interaction" = as.factor(interaction)) %>%
  pivot_wider(names_from=interaction, values_from = new_LRP)

correlation <- cor(quintum_raw[,-1], method = "pearson")
min(correlation)
sort(correlation %>% unique())


