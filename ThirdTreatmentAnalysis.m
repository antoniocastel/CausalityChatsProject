%% Data for last treatment 

LOS_in_website_before_invorRequest=...
chat_start_time-start_time;

%todos son Kab
sum(LOS_in_website_before_invorRequest<0)
sum(outcome==4); 
LOS_in_website_before_invorRequest(outcome==4,1)=...
invitation_submit_time(outcome==4,1)-start_time(outcome==4,1);

sum(LOS_in_website_before_invorRequest<0) %0
sum(LOS_in_website_before_invorRequest==0) %3

%invite type treatment
InvT=invite_type;
InvT(invite_type==2,1)=1;
InvT(invite_type==1,1)=0;


%with the variables I need 

CT1=[id_session,invite_type, region, city,country,continent, user_os, browser, score,...
     Invitation_Acep_Day_of_week,Invitation_Acep_Hour,...
     LOS_in_website_before_invorRequest,InvT,...
     Y];
CT1=string(CT1);

CT2={'id_session','invite_type','region','city','country','continent','user_os', 'browser', 'score',...
     'Arrival_Day_of_week','Arrival_Hour',...
     'LOS_in_website_before_invorRequest','InvT',...
     'Y'};
 
CT2=string(CT2);
CT3=[CT2;CT1];
writematrix(CT3,['DataForThirdTreatment.csv'])


save('DataChatsJan16.mat')

