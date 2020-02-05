#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

library(ggplot2)
library(ggsci)
library(latex2exp)
library(extrafont)
library(reshape2)
library(optparse)
font_install('fontcm')
par(family = "LM Roman 10")
loadfonts()

# read data to dataframe
stats <- read.csv("./data/USElectionDebates/corpus/stats_tokens.csv",
                  stringsAsFactors=FALSE)
debates <- gsub(".*\\_","",list.files("./data/USElectionDebates/",pattern="txt$"))
debates <- sort(as.numeric(gsub("\\.txt$","",debates)))
stats[,1] <- sapply(stats[,1],function(x) debates[x+1])

############################
# token distribution by year
############################

# aggregate data
agg <- aggregate(stats,by=list(stats$debate),FUN=sum)
agg <- agg[-c(2)]
names(agg)[1] <- "Year"
names(agg)[c(2,3,4)] <- c("None","Claim","Premise")
agg <- melt(agg,id.vars ="Year")
# create file
pdf(paste0("./img/","global.pdf"), width=11, height=12)
# make ggplot object
plot <- ggplot(agg,aes(x=factor(Year),y=value,fill=variable)) +
  geom_bar(stat="identity", color="black", size = 0.3)+xlab("\nYear")+ylab("Token Count\n") +
  theme_bw() +
  theme(text = element_text(size=25, family="CM Roman"),
        axis.text.x = element_text(angle = 90, hjust = 1),
        legend.text = element_text(size=25),
        legend.title = element_text(size=25,face = "bold"),
        legend.key = element_rect(colour = "lightgray", fill = "white"),
        plot.title = element_text(hjust=0.5)) +
  ggtitle("Token Type Distribution by Year") +
  scale_fill_npg(name="Token Type",alpha=0.8)
# process
print(plot)
dev.off()
# embed latex CM modern
embed_fonts(paste0("./img/","global.pdf"),
            outfile=paste0("./img/","global.pdf"))

##############################
# token distribution by length
##############################

stats$total <- rowSums(stats[,c(2:4)])
stats$bin <- cut(stats$total,seq(0,max(stats$total),100),dig.lab=5)
agg <- aggregate(stats[c("N","C","P")],by=list(stats$bin),FUN=sum)
names(agg)[1] <- "bin"
names(agg)[c(2,3,4)] <- c("None","Claim","Premise")
levels(agg$bin)
agg <- rbind(agg,c("(1000,1100]",0,0,0))
agg <- rbind(agg,c("(1200,1300]",0,0,0))
agg <- rbind(agg,c("(1300,1400]",0,0,0))
agg <- melt(agg,id.vars ="bin")
agg$value <- as.numeric(agg$value)
# create file
pdf(paste0("./img/","global_length.pdf"), width=11, height=12)
# make ggplot object
plot <- ggplot(agg,aes(x=bin,y=value,fill=variable)) +
  geom_bar(stat="identity", color="black", size = 0.3)+xlab("\nBinned Utterance Length")+ylab("Token Count\n") +
  theme_bw() +
  theme(text = element_text(size=25, family="CM Roman"),
        axis.text.x = element_text(angle = 90, hjust = 1),
        legend.text = element_text(size=25),
        legend.title = element_text(size=25,face = "bold"),
        legend.key = element_rect(colour = "lightgray", fill = "white"),
        plot.title = element_text(hjust=0.5)) +
  ggtitle("Token Type Distribution by Utterance Length") +
  scale_fill_npg(name="Token Type",alpha=0.8)
# process
print(plot)
dev.off()
# embed latex CM modern
embed_fonts(paste0("./img/","global_length.pdf"),
            outfile=paste0("./img/","global_length.pdf"))

#########################################
# token distribution by truncated length
#########################################

stats$type = "Unfiltered"
to_add <- stats
to_add[which(to_add[,5] > 128),c(2,3,4)] = 0
to_add$type <- "Filtered"
stats <- rbind(stats,to_add)
agg <- aggregate(stats[c("N","C","P")],by=list(stats$bin,stats$type),FUN=sum)
agg <- rbind(agg,c("(1000,1100]","Filtered",0,0,0))
agg <- rbind(agg,c("(1200,1300]","Filtered",0,0,0))
agg <- rbind(agg,c("(1300,1400]","Filtered",0,0,0))
agg <- rbind(agg,c("(1000,1100]","Unfiltered",0,0,0))
agg <- rbind(agg,c("(1200,1300]","Unfiltered",0,0,0))
agg <- rbind(agg,c("(1300,1400]","Unfiltered",0,0,0))
names(agg)[1] <- "bin"
names(agg)[2] <- "type"
names(agg)[c(3,4,5)] <- c("None","Claim","Premise")
agg <- melt(agg,id.vars =c("bin","type"))
agg[,2] <- factor(agg[,2],levels=c("Unfiltered","Filtered"))
levels(agg$type) <- c(TeX("Full corpus"),TeX("Pruned corpus \\[Sequence Length $\\leq$ 128\\]"))
agg$value <- as.numeric(agg$value)
# create file
pdf(paste0("./img/","global_length_trunc.pdf"), width=18, height=12)
# make ggplot object
plot <- ggplot(agg,aes(x=bin,y=value,fill=variable)) +
  geom_bar(stat="identity", color="black", size = 0.3)+xlab("\nBinned Utterance Length")+ylab("Token Count\n") +
  theme_bw() +
  theme(text = element_text(size=25),
        axis.text.x = element_text(angle = 90, hjust = 1),
        legend.text = element_text(size=25),
        legend.title = element_text(size=25,face = "bold"),
        legend.key = element_rect(colour = "lightgray", fill = "white"),
        plot.title = element_text(hjust=0.5)) +
  ggtitle("Token Type Distribution by Utterance Length") +
  scale_fill_npg(name="Token Type",alpha=0.8) +
  facet_wrap(~type,ncol=2,labeller = label_parsed)
# process
print(plot)
dev.off()
