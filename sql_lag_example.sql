--select wk_num, sales_amt_act, lag(sales_amt_act, 1, 0) over (order by wk_num)
--from 
--(select top (100) * from [dbo].[final_output] 
--where cust_list_code = 52887 and bu_size = 1015 and fcst_type = 'pos' order by wk_num) t


select fcst_type, wk_num, sales_amt_act, lag(sales_amt_act, 1, 0) over (partition by fcst_type order by wk_num)
from [dbo].[final_output] 
where cust_list_code = 52887 and bu_size = 1015 order by fcst_type, wk_num