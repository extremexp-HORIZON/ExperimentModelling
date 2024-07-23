COLUMNS_IOT = ['feat_orig_pkts_mean', 'feat_resp_pkts_mean',
                'feat_orig_bytes_mean', 'feat_resp_bytes_mean', 'feat_mean_duration',
                'feat_conn_state_dns_ratio', 'feat_conn_state_http_ratio',
                'feat_conn_state_ssl_ratio', 'feat_conn_state_dhcp_ratio',
                'feat_conn_state_unk_ratio', 'feat_conn_state_irc_ratio',
                'feat_method_dns_ratio', 'feat_method_http_ratio',
                'feat_method_ssl_ratio', 'feat_method_dhcp_ratio',
                'feat_method_unk_ratio', 'feat_method_irc_ratio',
                'feat_proto_tcp_ratio', 'feat_proto_udp_ratio', 
                'feat_proto_icmp_ratio', 'feat_private_conn_ratio']


COLUMNS_PHISHING = ['dns_interlog_time_0.001', 'dns_interlog_time_88.2912',
       'dns_interlog_time_176.5814', 'dns_interlog_time_264.87159999999994',
       'dns_interlog_time_353.16179999999997', 'ssl_interlog_time_0.001',
       'ssl_interlog_time_116.78700000000002',
       'ssl_interlog_time_233.57300000000004',
       'ssl_interlog_time_350.35900000000004',
       'ssl_interlog_time_467.14500000000004', 'http_interlog_time_0.001',
       'http_interlog_time_116.82900000000001', 'http_interlog_time_233.657',
       'http_interlog_time_350.485', 'http_interlog_time_467.313',
       'mean_interlog_time_dns_interlog_time',
       'std_interlog_time_dns_interlog_time',
       'mean_interlog_time_ssl_interlog_time',
       'std_interlog_time_ssl_interlog_time',
       'mean_interlog_time_http_interlog_time',
       'std_interlog_time_http_interlog_time', 'dns_protocol_tcp_ratio',
       'dns_protocol_udp_ratio', 'dns_common_tcp_ports_ratio',
       'dns_common_udp_ports_ratio', 'dns_rcode_noerror_ratio',
       'dns_rcode_nxdomain_ratio', 'dns_authoritative_ans_ratio',
       'dns_recursion_desired_ratio', 'dns_rejected_ratio',
       'dns_truncation_ratio', 'dns_mean_TTL', 'dns_len_TTL',
       'dns_qtype_used_ratio', 'dns_qtype_obsolete_ratio',
       'dns_non_reserved_srcport_ratio', 'dns_non_reserved_dstport_ratio',
       'dns_usual_dns_srcport_ratio', 'dns_usual_dns_dstport_ratio',
       'dns_shorturl_ratio', 'dns_compromised_domain_ratio',
       'dns_compromised_dstip_ratio', 'dns_socialmedia_ratio',
       'ssl_version_ratio_v10', 'ssl_version_ratio_v20',
       'ssl_version_ratio_v30', 'ssl_established_ratio',
       'ssl_compromised_dst_ip_ratio', 'ssl_resumed_ratio',
       'ssl_validation_status_ratio', 'ssl_curve_standard_ratio',
       'ssl_last_alert_ratio', 'http_request_body_len_ratio',
       'http_response_body_len_ratio', 'http_method_get_ratio',
       'http_method_post_ratio', 'http_method_head_ratio',
       'http_method_put_ratio', 'http_method_delete_ratio',
       'http_status_200_ratio', 'http_status_300_ratio',
       'http_status_400_ratio', 'http_common_ua_ratio',
       'http_private_con_ratio', 'http_compromised_dstip_ratio',
       'http_version_obsolete_ratio', 'http_version_used_ratio',
       'http_referrer_host_ratio', 'smtp_in_ratio', 'smtp_in_mean_hops',
       'smtp_subject_num_words', 'smtp_subject_num_characters',
       'smtp_subject_richness', 'smtp_subject_in_ratio_phishing_words',
       'smtp_in_is_reply', 'smtp_in_is_forwarded', 'smtp_in_is_normal',
       'smtp_in_is_spam', 'smtp_in_files', 'smtp_in_hazardous_extensions',
       'non_working_days_dns', 'non_working_days_http',
       'non_working_days_ssl', 'non_working_hours_dns',
       'non_working_hours_http', 'non_working_hours_ssl']
