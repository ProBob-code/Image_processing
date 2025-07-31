import os, requests, datetime, json, re
from classes.dbconnection import DbConnection
from rabbitmq import RabbitMQ
from helper import getConfigInfo
from common import generateRandom, RemoveFile, CurlGetJson, CurlPostPlain, sendMediaLogs

RESTRICTED_CATEGORY_API = 'http://192.168.131.170/cs/v1/category/api/check-restricted-category'
HOT_LEAD_POINT = 'http://192.168.131.98/api/leads/add'

class Utility:
    def __init__(self):
        pass
    
    def clean_string(self, string):
        string = string.replace('&', '-and-')
        string = string.replace(' ', '-')
        string = string.lower()
        string = re.sub(r'[^a-z0-9]', '-', string, flags=re.IGNORECASE)
        string = re.sub(r'-+', '-', string)
        return string
 
    def GetCompanyDetails(self, docid):
        returned_content = CurlGetJson(f"{getConfigInfo('COMPANY_DETAILS.url')}?docid={docid}&case=content_service&bprocess=common")
        # returned_content = json.loads(response)
        docid = str.upper(docid)

        if docid in returned_content:
            data = returned_content[docid]
            company_details = {}
            company_details['docid'] = docid
            company_details['parentid'] = data.get('parentid')
            company_details['company_name'] = self.clean_string(data.get('companyname', ''))
            company_details['area'] = self.clean_string(data.get('area', ''))
            company_details['city'] = (self.clean_string(data.get('city')) 
                                       if data.get('city') else self.clean_string(data.get('data_city', '')))
            company_details['data_city'] = (self.clean_string(data.get('data_city', '').lower()) 
                                            if data.get('data_city') else self.clean_string(data.get('city', '').lower()))
            company_details['hot_category'] = (self.clean_string(data.get('hot_cat_info', {}).get('name')) 
                                               if data.get('hot_cat_info') else '')
            company_details['catidlineage'] = data.get('catidlineage', '')
            company_details['latitude'] = data.get('latitude', '0')
            company_details['longitude'] = data.get('longitude', '0')
            company_details['pincode'] = data.get('pincode', '0')
            company_details['paidstatus'] = data.get('paidstatus', '0')
            company_details['price_range'] = data.get('price_range', '0')
            company_details['email_feedback'] = data.get('email_feedback', '')
            company_details['shorturl'] = data.get('shorturl', '')
            company_details['new_catidlineage'] = data.get('new_catidlineage', '')
            company_details['business_flag_abbr'] = data.get('business_flag_abbr', '')
            company_details['business_tag'] = data.get('business_tag', '')
            company_details['d_web_review'] = data.get('d_web_review', '0')
            company_details['data_city_ori'] = data.get('data_city', '')
            company_details['tag_info'] = data.get('tag_info', '')
            company_details['b2b_flag'] = data.get('b2b_flag', '0')
            return company_details
        else:
            return None
    
    def RemoveDataQueue(file_path):
        rabbitMq = RabbitMQ()
        try:
            path = dict()
            path["path"] = file_path
            
            # rabbitMq = RabbitMQ()
            queueData = dict()
            queueHost = getConfigInfo('rabbitmq_server1')
            queueHost["host"] = "/"
            
            queueData["credentials"] = queueHost
            queueData["DATA"] = path
            queueData["queue_name"] = "DELETE_PROCESSED_DATA"
            
            response = rabbitMq.postQueue(queueData)
            # os.remove(file_path)
            RemoveFile(file_path)
            print(f"Removed file by queue: {file_path}")
            return True
        except Exception as e:
            print(f"Error pushing file to DELETE_PROCESSED_DATA queue: {e}")
            return False
    
    def FetchCatalogueId(self, docid, company_name):
        db_connection = DbConnection()
        try:
            if docid != "":
                dbconn = db_connection.db_connect_live()
                dbcursor = dbconn.cursor(prepared=True)
                query = "SELECT catalogue_id FROM tbl_catalogue_main WHERE docid = %s AND delete_flag = 0 AND is_general = '1' LIMIT 1"
                dbcursor.execute(query, (docid,))
                row = dbcursor.fetchone()
                if row:
                    return row[0]
                else:
                    # Insert a new catalogue entry if not found
                    random_catalog_key = generateRandom(15)
                    catalogue_ins_sql = """
                        INSERT INTO tbl_catalogue_main 
                        SET docid = %s, company_name = %s, catalogue_name = 'General', 
                            is_general = '1', approved = '1', create_date = NOW(), 
                            module_type = '3', random_catalog_key = %s, created_by = 'backend_process'
                    """
                    dbcursor.execute(catalogue_ins_sql, (docid, company_name, random_catalog_key))
                    dbconn.commit()
                    catalogue_id_sql = """
                        SELECT catalogue_id FROM tbl_catalogue_main 
                        WHERE docid = %s AND is_general = 1 AND random_catalog_key = %s
                    """
                    dbcursor.execute(catalogue_id_sql, (docid, random_catalog_key))
                    catalogue_id_row = dbcursor.fetchone()
                    if catalogue_id_row:
                        return catalogue_id_row[0]
                    else:
                        print("Catalogue ID not found after insertion.")
                        return 0
            else:
                return 0
        except Exception as e:
            print(f"Error found {e}")
            return 0
        
    def getMainCityCircle(self, city):
        db_connection = DbConnection()
        main_city = 'remote'
        try:
            if city != "":
                dbconn = db_connection.db_connect_slave()
                dbcursor = dbconn.cursor(prepared=True)
                query = "SELECT maincity FROM tbl_catalog_citywise_breakup WHERE remote_city = %s LIMIT 1"
                dbcursor.execute(query, (city,))
                row = dbcursor.fetchone()
                if row:
                    main_city = row[0]
                dbcursor.close()
        except Exception as e:
            print(f"Error found {e}")
        return main_city
    
    def getContractCityCircle(self, docid, city=''):
        city = city.strip()
        docid = docid.lower()
        db_connection = DbConnection()
        if city:
            return self.getMainCityCircle(city)
        else:
            db_con = db_connection.db_connect_slave()
            # Query for docid in tbl_parent_status_web_consolidate
            consolidate_docid_sql = "SELECT data_city FROM tbl_parent_status_web_consolidate WHERE docid = %s"
            cursor = db_con.cursor()
            cursor.execute(consolidate_docid_sql, (docid,))
            consolidate_docid_row = cursor.fetchone()

            if consolidate_docid_row:
                cursor.close()
                return self.getMainCityCircle(consolidate_docid_row[0])
            else:
                # Extract parentid from docid
                parentid = docid[docid.find('p'):]
                consolidate_parentid_sql = "SELECT data_city FROM tbl_parent_status_web_consolidate WHERE parentid = %s"
                cursor.execute(consolidate_parentid_sql, (parentid,))
                consolidate_parentid_row = cursor.fetchone()

                if consolidate_parentid_row:
                    cursor.close()
                    return self.getMainCityCircle(consolidate_parentid_row[0])
                else:
                    data_city = ''
                    # Extract stdcode before 'P' in docid
                    stdcode = docid[:docid.upper().find('P')] if 'P' in docid.upper() else ''
                    db_con_meidc = db_connection.db_connect_meidc()
                    cursor_meidc = db_con_meidc.cursor()
                    data_city_sql = "SELECT data_city FROM city_master WHERE stdcode = %s"
                    cursor_meidc.execute(data_city_sql, (stdcode,))
                    data_city_row = cursor_meidc.fetchone()

                    if data_city_row:
                        cursor_meidc.close()
                        parentid_city_sql = "SELECT city, data_city FROM tbl_parent_status_web_consolidate WHERE parentid = %s AND data_city = %s"
                        cursor.execute(parentid_city_sql, (parentid, data_city_row[0]))
                        parentid_city_row = cursor.fetchone()

                        if parentid_city_row:
                            data_city = parentid_city_row[1]
                            cursor.close()
                            return self.getMainCityCircle(data_city)
                        else:
                            cursor.close()
                            return self.getMainCityCircle(data_city)
                    else:
                        cursor.close()
                        return self.getMainCityCircle(data_city)
    
    @staticmethod
    def check_restricted_category(data=None):
        db_connection = DbConnection()
        db_con = db_connection.db_connect_live()
        if data is None:
            data = {}
        
        cat_display_flag = 0
        cat_list_details = []
        
        if (data.get('new_catidlineage') and len(data.get('new_catidlineage', {})) > 0 and len(data.get('new_catidlineage', {}).get('val', [])) > 0):
            for cat_value in data['new_catidlineage']['val']:
                cat_list_details.append(cat_value[4])
            
            natid = ','.join(map(str, cat_list_details))
            # print(f"Combined catdId : {natid}")
            
            restricted_category_api_res = CurlGetJson(f"{RESTRICTED_CATEGORY_API}?categories={natid}")
            # restricted_category_api_res = json.loads(response)
            print(f"restricted_category_api_res : {restricted_category_api_res}")
            
            if restricted_category_api_res.get('data', {}).get('restricted_category') == 1:
                cat_display_flag = 1
                # content_type = data.get('content', '')
                
                # if content_type == 'photo':
                #     # updt_qry = f"""
                #     #     UPDATE tbl_catalogue_details 
                #     #     SET approved=0, modified_by='restriction_cat', modified_date=NOW() 
                #     #     WHERE docid='{data.get('docid')}' 
                #     #     AND random_catalog_key='{data.get('rand')}' 
                #     #     AND approved IN (1,2)
                #     # """
                #     # if data.get('module_type') in [3, 8, 9, 10, 11, 13, 43]:
                #     #     updt_qry = f"""
                #     #         UPDATE tbl_catalogue_details 
                #     #         SET approved=2, modified_by='restriction_cat', modified_date=NOW() 
                #     #         WHERE docid='{data.get('docid')}' 
                #     #         AND random_catalog_key='{data.get('rand')}' 
                #     #         AND approved IN (1,2)
                #     #     """
                
                # elif content_type == 'video':
                #     # updt_qry = f"""
                #     #     UPDATE tbl_video_details 
                #     #     SET approved=0, modified_by='restriction_cat', modified_date=NOW() 
                #     #     WHERE docid='{data.get('docid')}' 
                #     #     AND random_key='{data.get('rand')}' 
                #     #     AND approved IN (1,2) 
                #     #     AND video_tag='{data.get('video_tag')}'
                #     # """
                #     # MYSQL TO MONGO
                #     mysql_to_mongo = {
                #         'action': 'update',
                #         'fields': {
                #             'approved': '0',
                #             'modified_by': 'restriction_cat',
                #             'modified_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                #         },
                #         'where_con': [
                #             {'key': 'docid', 'value': data.get('docid'), 'op': '$eq'},
                #             {'key': 'random_key', 'value': data.get('rand'), 'op': '$eq'},
                #             {'key': 'approved', 'value': ['1', '2'], 'op': '$in'},
                #             {'key': 'video_tag', 'value': data.get('video_tag'), 'op': '$eq'}
                #         ],
                #         'file': 'joinfree_docid_update'
                #     }
                #     # Placeholder for MongoDB queue operation
                #     # video_mysql_to_mongo(mysql_to_mongo, "VIDEO_MONGO_QUEUE")
                
                # elif content_type == 'video_amz':
                #     updt_qry = f"""
                #         UPDATE tbl_video_details 
                #         SET approved=0, modified_by='restriction_cat', modified_date=NOW() 
                #         WHERE docid='{data.get('docid')}' 
                #         AND ref_id='{data.get('rand')}' 
                #         AND approved IN (1,2) 
                #         AND video_tag='{data.get('video_tag')}'
                #     """
                #     # MYSQL TO MONGO
                #     mysql_to_mongo = {
                #         'action': 'update',
                #         'fields': {
                #             'approved': '0',
                #             'modified_by': 'restriction_cat',
                #             'modified_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                #         },
                #         'where_con': [
                #             {'key': 'docid', 'value': data.get('docid'), 'op': '$eq'},
                #             {'key': 'ref_id', 'value': data.get('rand'), 'op': '$eq'},
                #             {'key': 'approved', 'value': ['1', '2'], 'op': '$in'},
                #             {'key': 'video_tag', 'value': data.get('video_tag'), 'op': '$eq'}
                #         ],
                #         'file': 'joinfree_docid_update'
                #     }
                #     # Placeholder for MongoDB queue operation
                #     # video_mysql_to_mongo(mysql_to_mongo, "VIDEO_MONGO_QUEUE")
                
                # else:
                #     updt_qry = f"""
                #         UPDATE tbl_catalogue_details 
                #         SET approved=0, modified_by='restriction_cat', modified_date=NOW() 
                #         WHERE docid='{data.get('docid')}' 
                #         AND random_catalog_key='{data.get('rand')}' 
                #         AND approved IN (1,2)
                #     """
                
                # Execute update query (assuming db_con is a valid database connection)
                try:
                    # db_con.execute(updt_qry)
                    # db_con.commit()
                    
                    # Insert into restricted category details
                    ins_qry = f"""
                        INSERT INTO tbl_catalogue_restricted_category_details 
                        SET docid='{data.get('docid')}', 
                        create_date=NOW(), 
                        random_catalog_key='{data.get('rand')}'
                    """
                    db_con.execute(ins_qry)
                    db_con.commit()
                
                except Exception as e:
                    print(f"Database error: {e}")
                    db_con.rollback()
        
        return cat_display_flag
    
    def generate_hot_lead(docid, module_type, paidstatus, approved, upload_by, obj_utility, obj_dbcon):
        exclude_module_types = [3, 4, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                                22, 25, 26, 28, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40]

        if paidstatus == 0 and approved == 2 and module_type not in exclude_module_types:
            hot_lead_trigger = True

            # Check if upload_by looks like a mobile (10 digit number)
            if re.match(r"^[0-9]{10}$", str(upload_by)):
                external_check_api = f"https://win-stg.justdial.com/01march2019/checkDcNumber.php?type=dc&mobile={upload_by}"
                try:
                    external_api_res = json.loads(CurlGetJson(external_check_api))
                except Exception as e:
                    print("Error calling external check API:", e)
                    external_api_res = {}
                if external_api_res.get("success") == 1:
                    print("Number Found, Hot Lead is not required")
                    hot_lead_trigger = False
                else:
                    print("Comparing number with operations team")
                    emp_check_sql = f"SELECT * FROM tbl_emp_details WHERE Dept='Operations' AND mobile_number='{upload_by}' LIMIT 1"
                    emp_res = obj_dbcon.execVLC236(emp_check_sql)
                    # Assume execVLC236 returns an iterable (list of rows)
                    if emp_res and len(emp_res) > 0:
                        hot_lead_trigger = False

            if hot_lead_trigger:
                try:
                    req_data = {
                        "g_queue": 1,
                        "docid": docid,
                        "campaign_name": "useredit",
                        "platform": "uservideoupload",
                        "data_source": "Non Verified Edit Listing"
                    }
                    hot_response = CurlPostPlain(HOT_LEAD_POINT, req_data)

                    logs_data = {
                        "id": docid,
                        "publish": "MEDIA",
                        "route": "HOT LEAD CALL",
                        "user_id": upload_by,
                        "critical": 1,
                        "msg": "hotlead api call in video",
                        "query": json.dumps({
                            "post_data": req_data,
                            "post_res": hot_response
                        })
                    }
                    sendMediaLogs(docid, logs_data, 'HOT LEAD CALL', 'hotlead api call in video', upload_by)
                    print(hot_response)
                except Exception as e:
                    print("Hot lead API call failed:", e)