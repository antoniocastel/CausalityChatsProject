
NumberofAssignedwhenAssigned=zeros(length(id_rep),1);

%number of assigned 
for i=1:length(id_session)
    numberAg=id_rep(i,1);
    
    if numberAg==1
        %no assigment
        NumberofAssignedwhenAssigned(i,1)=0;
        continue
    end
    
    DumStart=...
        queue_exit_time(i,1);
    DumEnd  =...
        chat_end_time(i,1);
    
    %that sessh
    counter=1;
    
    for j=1:length(id_session)
        if(id_session(j,1)==id_session(i,1))
            continue
        end
        
        DumStart2=...
        queue_exit_time(j,1);
        DumEnd2  =...
        chat_end_time(j,1);
        if (DumStart>=DumStart2) && (DumStart<=DumEnd2) 

        %if (DumStart2>=DumStart) && (DumEnd2<=DumEnd) 
            %who served him?
            if   id_rep(j,1)==numberAg
            %same as the i session
            counter=counter+1;
            end 

        end
    
    end
    if(counter>3)
       counter=3;
    end
    NumberofAssignedwhenAssigned(i,1)=counter;
end 


sum(engagement_skill==4)/length(engagement_skill)

UniqueAgent=unique(id_rep);
%first one is no agent
UniqueAgent=UniqueAgent(2:end,1);
SkillofthatAgent=zeros(length(UniqueAgent),1);
for i=1:length(UniqueAgent)
   Uni=UniqueAgent(i,1);
   indexx=find(id_rep==Uni,1,'first');
   SkillofthatAgent(i,1)=engagement_skill(indexx,1);
   
end

sum(SkillofthatAgent==1)/length(SkillofthatAgent)
sum(SkillofthatAgent==2)/length(SkillofthatAgent)
sum(SkillofthatAgent==3)/length(SkillofthatAgent)

mean(score)
std(score)

uni.os=unique(user_os); %17 values
sum(user_os==uni.os(1,1))/length(user_os) %NONE
user_os_desc(find(user_os==uni.os(1,1),1,'first'),1) 

sum(user_os==uni.os(2,1))/length(user_os) %LINUX
user_os_desc(find(user_os==uni.os(2,1),1,'first'),1) 


sum(user_os==uni.os(3,1))/length(user_os) % 0.2758
find(user_os==5,1,'first') %MAC os
sum(user_os==uni.os(4,1))/length(user_os)
user_os_desc(find(user_os==uni.os(4,1),1,'first'),1) 

sum(user_os==uni.os(5,1))/length(user_os)
user_os_desc(find(user_os==uni.os(5,1),1,'first'),1) 

sum(user_os==uni.os(6,1))/length(user_os)
user_os_desc(find(user_os==uni.os(6,1),1,'first'),1) 

sum(user_os==uni.os(7,1))/length(user_os)
user_os_desc(find(user_os==uni.os(7,1),1,'first'),1) 

sum(user_os==uni.os(8,1))/length(user_os)
user_os_desc(find(user_os==uni.os(8,1),1,'first'),1) 

sum(user_os==uni.os(9,1))/length(user_os)
user_os_desc(find(user_os==uni.os(9,1),1,'first'),1) 


sum(user_os==uni.os(10,1))/length(user_os) %0.767
find(user_os==uni.os(10,1),1,'first') %
user_os_desc(66,1) %Mac OS X Yosemite 

sum(user_os==uni.os(3,1))+sum(user_os==uni.os(4,1))+sum(user_os==uni.os(5,1))+sum(user_os==uni.os(6,1))+sum(user_os==uni.os(7,1))+sum(user_os==uni.os(8,1))+sum(user_os==uni.os(9,1))+sum(user_os==uni.os(10,1))
9676/length(user_os) %41.38

sum(user_os==uni.os(11,1))/length(user_os)
user_os_desc(find(user_os==uni.os(11,1),1,'first'),1) 

sum(user_os==uni.os(12,1))/length(user_os)
sum(user_os==uni.os(13,1))/length(user_os)

sum(user_os==uni.os(14,1))/length(user_os) %0.2936
find(user_os==uni.os(14,1),1,'first') %
user_os_desc(4,1) %Windows 7

sum(user_os==uni.os(15,1))/length(user_os)

sum(user_os==uni.os(16,1))/length(user_os)
find(user_os==uni.os(16,1),1,'first') %0.0766
user_os_desc(14,1) %Windows 8.1

sum(user_os==uni.os(17,1))/length(user_os) %0.1482
find(user_os==uni.os(17,1),1,'first') %
user_os_desc(9,1) %Windows 10

sum(user_os==uni.os(11,1))+sum(user_os==uni.os(12,1))+sum(user_os==uni.os(13,1))+sum(user_os==uni.os(14,1))+sum(user_os==uni.os(15,1))+sum(user_os==uni.os(16,1))+sum(user_os==uni.os(17,1))
%13061/length(user_os) %55.86


%uni.bw=unique(browser); %85 values
%unique(browser_desc)

%% SCORE distribution
max(score) %0.4074

intervalWidth=.01;

xS = 0:intervalWidth:max(score);
ncountS = histc(score,xS);
relativefreqS = ncountS/length(score);

%filter smaller than .2;
xSf=xS(xS<=.2);
relativefreqSf=relativefreqS(xS<=.2);

%esta grafica tiene 30% de los datos en menos de 20 sec.
%no me piace la corto
bar(xSf-intervalWidth/2, relativefreqSf,1)
xlim([min(xSf) max(xSf)])
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Score') 
ylabel('Frequencies')
%HistogramScore.fig

%% browser

uni.browser=unique(browser); %85 levels
uni.brow.ncount = histc(browser,uni.browser);
uni.brow.freq = uni.brow.ncount/length(browser);
uni.brow.freqS=sort(uni.brow.freq,'descend');

bar(uni.browser, uni.brow.freq,1)
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Score') 
ylabel('Frequencies')

[a1,a2]=max(uni.brow.freq)
%0.30+0.15+.07 =0.58
browser_desc(a2,1)
%27
%Chrome 47.X

%0.23
browser_desc(83,1)
%Firefox 36.x 

%0.15
browser_desc(69,1)
%tambien Chrome 47.x

browser_desc(62,1)
%tambien Chrome 47.x

browser_desc(82,1)
%Internet Explorer 7.0 %


%% continent 

uni.conti=unique(continent); %9
uni.contin.ncount = histc(continent,uni.conti);
uni.contin.freq = uni.contin.ncount/length(continent);

xs=1:9;
intervalWidth=1;
bar(xs-intervalWidth/2, uni.contin.freq,1)
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Continent') 
ylabel('Frequencies')

%0.84 continent 3 
%0.12 contuent 4 
%0.013 continet 2

%% region 

uni.reg=unique(region); %491
uni.regi.ncount = histc(region,uni.reg);
uni.regi.freq = uni.regi.ncount/length(region);
uni.regi.freqS=sort(uni.regi.freq,'descend');

xs=1:20;
intervalWidth=1;
bar(xs-intervalWidth/2, uni.regi.freqS(1:20,1),1)
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Region') 
ylabel('Frequencies')

%0.84 continent 3 
%0.12 contuent 4 
%0.013 continet 2



%% country 

uni.pais=unique(country); %108
uni.paisi.ncount = histc(country,uni.pais);
uni.paisi.freq = uni.paisi.ncount/length(country);
uni.paisi.freqS=sort(uni.paisi.freq,'descend');

%0.81 one country %12% other country %6 other country

xs=1:20;
intervalWidth=1;
bar(xs-intervalWidth/2, uni.regi.freqS(1:20,1),1)
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Region') 
ylabel('Frequencies')

%0.84 continent 3 
%0.12 contuent 4 
%0.013 continet 2

%% city 

uni.cit=unique(city); %2487
uni.citi.ncount = histc(country,uni.cit);
uni.citi.freq = uni.citi.ncount/length(city);
uni.citi.freqS=sort(uni.citi.freq,'descend');

%82 one city %12% other city 

%0.81 one country %12% other country %6 other country


%% Rho at arrival distribution
max(Rho) %15.52
Rhonozeros=Rho(Rho>0);
intervalWidth=.5;
mean(Rho) %1.49
std(Rho)  %1.82
xS = 0:intervalWidth:max(Rhonozeros);
ncountS = histc(Rhonozeros,xS);
relativefreqS = ncountS/length(Rhonozeros);

%filter smaller than 4;
xSf=xS(xS<=4);
relativefreqSf=relativefreqS(xS<=4);

%esta grafica tiene 30% de los datos en menos de 20 sec.
%no me piace la corto
bar(xSf-intervalWidth/2, relativefreqSf,1)
xlim([min(xSf) max(xSf)])
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('\rho','Fontweight','bold') 
ylabel('Frequencies')
%HistogramScore.fig


%% 'Invitation_Acep_Day_of_week','Invitation_Acep_Hour'
%saturday is day 5
%week starts on a monday with the 0


uni.dayWeek=unique(Invitation_Acep_Day_of_week); %7
uni.dayWeeki.ncount = histc(Invitation_Acep_Day_of_week,uni.dayWeek);
uni.dayWeeki.freq = uni.dayWeeki.ncount/length(Invitation_Acep_Day_of_week);

xs=1:7;
intervalWidth=1;
bar(xs-intervalWidth/2, uni.dayWeeki.freq,1)
xlim([min(0) max(7)])
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Day of the week of arrival') 
ylabel('Frequencies')

%most of the days have the same load
%but the arrivals peak during friday-sunday

%% Hour of Acceptance of invo
uni.Hour=unique(Invitation_Acep_Hour); %7-21
uni.Houri.ncount = histc(Invitation_Acep_Hour,uni.Hour);
uni.Houri.freq = uni.Houri.ncount/length(Invitation_Acep_Hour);

xs=uni.Hour(2:end,1);
intervalWidth=1;
bar(xs-intervalWidth/2, uni.Houri.freq(2:end,1),1)
xlim([7 max(21)])
xticks([7:22])
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Hour of arrival') 
ylabel('Frequencies')

%% Wait time Histogram

max(queue_sec/60) %7695.72000000000/60 %128 min
mean(queue_sec/60) %1.15 min
std(queue_sec/60) %2.63
intervalWidth=.5; %30 sec

xS = 0:intervalWidth:max(queue_sec/60);
ncountS = histc(queue_sec/60,xS);
relativefreqS = ncountS/length(queue_sec/60);

%filter smaller than 200/60=3.33 min;
xSf=xS(xS<=4.5);
relativefreqSf=relativefreqS(xS<=4.5);

%no me piace la corto
figure
bar(0:0.5:max(queue_sec/60), relativefreqS,1)
xlim([0 max(4.5)])
xticks([0:0.5:4.5])
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Outer Wait (min)') 
ylabel('Frequencies')
%HistogramScore.fig

bar(xS-intervalWidth/2, relativefreqS,1)
xlim([0 4.5])
xticks([7:22])
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Hour of arrival') 
ylabel('Frequencies')

%% AVERAGE 
mean([-0.003,0.003,-0.003,0.01]) %0.0018
mean([-0.01,-0.001,-0.009,0.02]) %0
mean([0.0003,-0.004,0.0002,0.008]) %0 %.0011
mean([-0.003,0.03,-0.003,0.02]) %0 %.011

mean([0.07,0.03,-0.007,0.07,-0.22]) %0 %-0.0114

mean([0.11,0.02,-0.01,0.11,-0.26]) %0 %-0.0114




%% HISTOGRAM INNER WAIT

max(other_time)
mean(other_time) %146.54
std(other_time/60)
%4554/60 %75 min
%interval width  20 seconds
intervalWidth=30/60;

xS = 0:intervalWidth:max(other_time/60);
ncountS = histc(other_time/60,xS);
relativefreqS = ncountS/length(other_time/60);

%xSf=xS(xS<=4.5);
%relativefreqSf=relativefreqS(xS<=4.5);

figure
bar(xS, relativefreqS,1)
xlim([0 6])
%xticks([7:22])
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Inner Wait (minutes)') 
ylabel('Frequencies')

%50% less than 61 seconds
%% HISTOGRAM customer response time

max(visitor_duration)
%4554/60 %75 min

mean(visitor_duration/60)
%227.51 seconds %3.80 min
std(visitor_duration/60)
%269.9 %4.43
%interval width  20 seconds
intervalWidth=30/60;

xS = 0:intervalWidth:max(visitor_duration/60);
ncountS = histc(visitor_duration/60,xS);
relativefreqS = ncountS/length(visitor_duration/60);

%xSf=xS(xS<=4.5);
%relativefreqSf=relativefreqS(xS<=4.5);

figure
bar(xS, relativefreqS,1)
xlim([0 6])
%xticks([7:22])
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Customer Response Time (minutes)') 
ylabel('Frequencies')

%% HISTOGRAM agent response time

max(agent_duration)
%4343/60 %72 min

mean(agent_duration/60)
%180.25 seconds
std(agent_duration/60)
%187.98
%interval width  20 seconds
intervalWidth=30/60;

xS = 0:intervalWidth:max(agent_duration/60);
ncountS = histc(agent_duration/60,xS);
relativefreqS = ncountS/length(agent_duration/60);

%xSf=xS(xS<=4.5);
%relativefreqSf=relativefreqS(xS<=4.5);

figure
bar(xS, relativefreqS,1)
xlim([0 6])
%xticks([7:22])
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Agent Response Time (minutes)') 
ylabel('Frequencies')



%% HISTOGRAM visitor number words

max(visitor_number_words)
%2744/60 %72 min

mean(visitor_number_words)
%67.97
std(visitor_number_words)
%79.65
%interval width  5 words
intervalWidth=10;

xS = 0:intervalWidth:max(visitor_number_words);
ncountS = histc(visitor_number_words,xS);
relativefreqS = ncountS/length(visitor_number_words);

%xSf=xS(xS<=4.5);
%relativefreqSf=relativefreqS(xS<=4.5);

figure
bar(xS, relativefreqS,1)
xlim([0 150])
%xticks([7:22])
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Customer Number of Words') 
ylabel('Frequencies')


%% HISTOGRAM agent number words

max(agent_number_words)
%3975

mean(agent_number_words)
%139.26
std(agent_number_words)
%138.19
%interval width  5 words
intervalWidth=10;

xS = 0:intervalWidth:max(agent_number_words);
ncountS = histc(agent_number_words,xS);
relativefreqS = ncountS/length(agent_number_words);

%xSf=xS(xS<=4.5);
%relativefreqSf=relativefreqS(xS<=4.5);

figure
bar(xS, relativefreqS,1,'EdgeColor','k')
xlim([0 220])
%xticks([7:22])
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Agent Number of Words') 
ylabel('Frequencies')


%% HISTOGRAM total canned lines

max(total_canned_lines)
%19

mean(total_canned_lines)
%1.38
std(total_canned_lines)
%1.45
%interval width  1 words
intervalWidth=1;

xS = 0:intervalWidth:max(total_canned_lines);
ncountS = histc(total_canned_lines,xS);
relativefreqS = ncountS/length(total_canned_lines);

%xSf=xS(xS<=4.5);
%relativefreqSf=relativefreqS(xS<=4.5);

figure
bar(xS, relativefreqS,1,'EdgeColor','k')
xlim([0 7])
%xticks([7:22])
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Total Canned Messages') 
ylabel('Frequencies')



%% HISTOGRAM sentiment 

max(average_sent)
%7

mean(average_sent)
%0.27
std(average_sent)
%0.89
%interval width  1 words
intervalWidth=.5;

xS = 0:intervalWidth:max(average_sent);
ncountS = histc(average_sent,xS);
relativefreqS = ncountS/length(average_sent);

%xSf=xS(xS<=4.5);
%relativefreqSf=relativefreqS(xS<=4.5);

figure
bar(xS, relativefreqS,1,'EdgeColor','k')
xlim([-1 3])
%xticks([7:22])
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Average Sentiment') 
ylabel('Frequencies')




%% HISTOGRAM sentiment 

max(NumberofAssignedwhenAssigned)
%3

mean(NumberofAssignedwhenAssigned)
%2.17
std(NumberofAssignedwhenAssigned)
%0.88
%interval width  1 words
intervalWidth=1;
%0.11
%0.13 0.26 0.51
xS = 1:intervalWidth:max(NumberofAssignedwhenAssigned);
ncountS = histc(NumberofAssignedwhenAssigned,1:1:3);
relativefreqS = ncountS/length(NumberofAssignedwhenAssigned);

%xSf=xS(xS<=4.5);
%relativefreqSf=relativefreqS(xS<=4.5);

figure
bar(xS, relativefreqS,1,'EdgeColor','k')
xlim([0 3.5])
%xticks([7:22])
set(gca,'FontSize',14) 
%set(gca, 'xtick', xSf2)
xlabel('Assigned Customers') 
ylabel('Frequencies')









