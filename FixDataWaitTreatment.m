
%%---
%this was before i fixed in it phyton
%known abandoments with wait<5 take out
%index=find(queue_sec==0);
%dummy=outcome(index,1);
%sum(dummy==4)
%222 todos son kab

%%%
sum(outcome==4) %1805
Total_Service_Duration(outcome==4) %only zeros

minarrivaltime=min(invitation_submit_time);

invitation_submit_time_Sformat=invitation_submit_time...
    -minarrivaltime;

chat_end_time_Sformat=chat_end_time...
     -minarrivaltime;


max(chat_end_time_Sformat) %2641859/3600 %734 hours
max(invitation_submit_time_Sformat) %2641407/3600 %734

ArrivalHourRate=zeros(735,1);
ClosureHourRate=zeros(735,1);

for i=1:735
   ArrivalHourRate(i,1)=...
       sum(   invitation_submit_time_Sformat>=(3600*(i-1)) ...
       &invitation_submit_time_Sformat<(3600*(i))   ) ;
   ClosureHourRate(i,1)=...
   sum(   chat_end_time_Sformat>=(3600*(i-1)) ...
       &chat_end_time_Sformat<(3600*(i))   ) ;

end


OfferedLoadHour=ArrivalHourRate./ClosureHourRate;

OfferedLoadHour(isnan(OfferedLoadHour))=0;
OfferedLoadHour(isinf(OfferedLoadHour))=0;


E_TS=zeros(length(ArrivalHourRate),1);
for i=1:735
  dummy=...
        invitation_submit_time_Sformat>=(3600*(i-1)) ...
       &invitation_submit_time_Sformat<(3600*(i)   ) ; 
   dummy2=Total_Service_Duration(dummy,1);
  E_TS(i,1)=mean(dummy2(dummy2>0));
end

E_TS(isnan(E_TS))=0;

NumAgents=zeros(735,1);
for i=1:735
  dummy=...
        invitation_submit_time_Sformat>=(3600*(i-1)) ...
       &invitation_submit_time_Sformat<(3600*(i)   ) ; 
  %number of unique agent id's in hour i
  NumAgents(i,1)=...
      length(unique(id_rep(dummy,1)));
end
NumofSlots=NumAgents*3;

NumAb=zeros(735,1);
NumCust=zeros(735,1);
for i=1:735
  dummy=...
        invitation_submit_time_Sformat>=(3600*(i-1)) ...
       &invitation_submit_time_Sformat<(3600*(i)   ) ; 
    dummy2=...
      outcome(dummy,1);
    NumAb(i,1)=...
        sum(dummy2==4);
    NumCust(i,1)=...
        sum(dummy2>0);
        
end

proportion_ab=NumAb./NumCust;
proportion_ab(isnan(proportion_ab))=0;


%Rho=ArrivalHourRate./(ClosureHourRate.*NumofSlots);
Rho= ( (ArrivalHourRate.*(1-proportion_ab)).*(E_TS/3600))...
    ./ NumofSlots;
Rho(isnan(Rho))=0;

E_InnerWait=zeros(length(ArrivalHourRate),1);
for i=1:735
  dummy=...
        invitation_submit_time_Sformat>=(3600*(i-1)) ...
       &invitation_submit_time_Sformat<(3600*(i)   ) ; 
   dummy2=inner_wait(dummy,1);
  E_InnerWait(i,1)=mean(dummy2(dummy2>0));
end
E_InnerWait(isnan(E_InnerWait))=0;


E_load_atarrival=...
    zeros(length(invitation_submit_time_Sformat),1);
NumofSlots_atarrival=...
    zeros(length(invitation_submit_time_Sformat),1);
E_innerW_atarrival=...
    zeros(length(invitation_submit_time_Sformat),1);
Rho_atarrival=...
    zeros(length(invitation_submit_time_Sformat),1);


for i=1:735
   E_load_atarrival...
           (invitation_submit_time_Sformat>=(3600*(i-1)) ...
       &    invitation_submit_time_Sformat<(3600*(i) ),1 ) =...
       OfferedLoadHour(i,1);
   
   NumofSlots_atarrival...
           (invitation_submit_time_Sformat>=(3600*(i-1)) ...
       &    invitation_submit_time_Sformat<(3600*(i) ),1 ) =...
       NumofSlots(i,1);
   
   E_innerW_atarrival...
           (invitation_submit_time_Sformat>=(3600*(i-1)) ...
       &    invitation_submit_time_Sformat<(3600*(i) ),1 ) =...
       E_InnerWait(i,1);
   
   Rho_atarrival...
           (invitation_submit_time_Sformat>=(3600*(i-1)) ...
       &    invitation_submit_time_Sformat<(3600*(i) ),1 ) =...
       Rho(i,1);
       
end

%with the variables I need 

C1=[id_session, queue_sec,invite_type, engagement_skill,target_skill,...
     region,city,country,continent,user_os, browser, score,...
     Invitation_Acep_Day_of_week,Invitation_Acep_Hour,...
     Rho_atarrival,WaitTreatment,Y];
C1=string(C1);

C2={'id_session', 'queue_sec','invite_type', 'engagement_skill','target_skill',...
     'region','city','country','continent','user_os', 'browser', 'score',...
     'Invitation_Acep_Day_of_week','Invitation_Acep_Hour',...
     'Rho_atarrival','WaitTreatment','Y'};
 
C2=string(C2);
C3=[C2;C1];
writematrix(C3,['DataForFirstTreatment.csv'])


save('DataChatsJan16.mat')

index=(queue_sec<120);
outcome_index = outcome(index,1);
queue_sec_index= queue_sec(index,1);
Teta=( sum(outcome<3 )/...
sum(queue_sec(outcome<3)) )...
-...
sum(outcome<4)/...
sum(queue_sec);

1/Teta %130.71

Teta=( sum(outcome_index<3 )/...
sum(queue_sec_index(outcome_index<3)) )...
-...
sum(outcome_index<4)/...
sum(queue_sec_index);

1/Teta %36.50

%% We assume that only customers with a $score>0.05$ 
%or that requested to chat wanted to purchase a ticket

sum(score>0.05) %3593

sum(invite_type==1) %11050

sum(score>0.05 | invite_type==1) %12343

%sanity Check
[score(score>0.05 | invite_type==1),invite_type(score>0.05 | invite_type==1)];

C1=[id_session, queue_sec,invite_type, engagement_skill,target_skill,...
     region,city,country,continent,user_os, browser, score,...
     Invitation_Acep_Day_of_week,Invitation_Acep_Hour,...
     Rho_atarrival,WaitTreatment,Y];
C1=C1(score>0.05 | invite_type==1,:);
C1=string(C1);

C2={'id_session', 'queue_sec','invite_type', 'engagement_skill','target_skill',...
     'region','city','country','continent','user_os', 'browser', 'score',...
     'Invitation_Acep_Day_of_week','Invitation_Acep_Hour',...
     'Rho_atarrival','WaitTreatment','Y'};
 
C2=string(C2);
C3=[C2;C1];
writematrix(C3,['DataForFirstTreatmentii.csv'])

save('DataChatsJan16.mat')


%% We assume that only customers with a $score>0.1$ 
%or that where invited when rho<1

sum(score>0.1) %1891

sum(invite_type==2&Rho_atarrival<=1) %241

sum( (score>0.1) | (invite_type==2&Rho_atarrival<=1) ) %2126

%sanity Check
[score( (score>0.1) | (invite_type==2&Rho_atarrival<=1) ),invite_type( (score>0.1) | (invite_type==2&Rho_atarrival<=1) )...
    Rho_atarrival( (score>0.1) | (invite_type==2&Rho_atarrival<=1) )];

C1=[id_session, queue_sec, invite_type, engagement_skill,target_skill,...
     region,city,country,continent,user_os, browser, score,...
     Invitation_Acep_Day_of_week,Invitation_Acep_Hour,...
     Rho_atarrival,WaitTreatment,Y];
C1=C1( ( (score>0.1) | (invite_type==2&Rho_atarrival<=1) ), :);
C1=string(C1);

C2={'id_session', 'queue_sec', 'invite_type', 'engagement_skill','target_skill',...
     'region','city','country','continent','user_os', 'browser', 'score',...
     'Invitation_Acep_Day_of_week','Invitation_Acep_Hour',...
     'Rho_atarrival','WaitTreatment','Y'};
 
C2=string(C2);
C3=[C2;C1];
writematrix(C3,['DataForFirstTreatmentiii.csv'])

save('DataChatsJan16.mat')

