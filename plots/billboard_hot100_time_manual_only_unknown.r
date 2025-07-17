library(ggplot2)

data=read.table("../data/billboard.hot100_manual_only_unknown.yearly.data")
data
pdf("billboard_hot100_manual_year_only_unknown.pdf", width=7)
ggplot(data, aes(x=V1, y=V2)) + geom_point() + geom_ribbon(aes(ymin=V3, ymax=V4), fill="blue", alpha=0.3) + xlim(1960,2025) + xlab("") + ylab("narrativity")
dev.off()
