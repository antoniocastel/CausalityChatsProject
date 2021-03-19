
NumberofAssigned=zeros(length(id_rep),1);

%number of assigned 
for i=1:length(id_session)
    numberAg=id_rep(i,1);
    
    if numberAg==1
        %no assigment
        NumberofAssigned(i,1)=0;
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
    
        if (DumStart2>=DumStart) && (DumEnd2<=DumEnd) 
            %who served him?
            if   id_rep(j,1)==numberAg
            %same as the i session
            counter=counter+1;
            end 

        end
    
    end
%     if(counter>3)
%        counter=3;
%     end
    NumberofAssigned(i,1)=counter;
end 

%%when assigned
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

a1=[2 10];
a2=[5 15];

2>5 %puede que emepzo dentro
2<5 %puede que empezo antes

10>15 %1-no esta dentro
%empieza despues del inicio y termina 
%despues del final 
          %6<5, 16<15 6<15 0 1 1
        %[6,16] si
           %16<5, 20<15 16<15 000
                %[16,20]no
    %2- empieza antes del incio y termina 
    %despues del final
                %4<5 16<15  4<15  1 0 1
        %[4,16] si
  %10<15
  
        %1-   6<5 8<15 6<15      01 1
        %[6,8] si
        
        %2  $4<5  8<15  &4<15    11 1
        %[4,8] si


%% new file for regression

%with the variables I need 

CA=[id_session, queue_sec,invite_type, engagement_skill,target_skill,...
     region,city,country,continent,user_os, browser, score,...
      id_rep ,  other_time  , other_lines ,  other_number_words ,  inner_wait ,  visitor_duration ,...
       agent_duration ,  total_duration ,  visitor_number_words ,  agent_number_words ,...
       total_number_words ,	 visitor_lines ,  agent_lines ,	 total_lines ,...
       total_canned_lines ,	  dur_dif ,	  average_sent ,  min_sent ,  max_sent ,  n_sent ,  n_sent_pos ,  n_sent_neg ,	 first_sent ,...
      last_sent,	id_rep_code,...
     Invitation_Acep_Day_of_week,Invitation_Acep_Hour,...
     NumberofAssigned, NumberofAssignedwhenAssigned,...
     Rho_atarrival,WaitTreatment,Y];
CA=string(CA);

CB={'id_session', 'queue_sec','invite_type', 'engagement_skill','target_skill',...
     'region','city','country','continent','user_os', 'browser', 'score',...
      'id_rep', 'other_time' ,'other_lines', 'other_number_words', 'inner_wait', 'visitor_duration',...
      'agent_duration', 'total_duration', 'visitor_number_words', 'agent_number_words',...
      'total_number_words',	'visitor_lines', 'agent_lines',	'total_lines',...
      'total_canned_lines',	 'dur_dif',	 'average_sent', 'min_sent', 'max_sent', 'n_sent', 'n_sent_pos', 'n_sent_neg',	'first_sent',...
      'last_sent',	'id_rep_code',...
     'Invitation_Acep_Day_of_week','Invitation_Acep_Hour',...
     'NumberofAssigned', 'NumberofAssignedwhenAssigned',...
     'Rho_atarrival','WaitTreatment','Y'};
 
CB=string(CB);
CC=[CB;CA];
writematrix(CC,['DataForRegression.csv'])


save('DataChatsJan16.mat')