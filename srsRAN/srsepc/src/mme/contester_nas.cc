#include "srsepc/hdr/mme/s1ap.h"
#include "srsepc/hdr/mme/nas.h"

#include <cmath>
#include <inttypes.h>
#include <iostream>
#include <stdbool.h>
#include <iomanip>
#include <signal.h>
#include <time.h>
#include <unistd.h>

#include <fstream>

#include "srsepc/hdr/mme/contester_nas.h"


namespace srsepc {

size_t sig_id0, sig_id1, sig_id2;
contester_nas*  contester_nas::m_instance       = NULL;
pthread_mutex_t contester_nas_instance_locker   = PTHREAD_MUTEX_INITIALIZER;
static bool     wait_for_response               = false;

contester_nas::contester_nas()
{
  sig_id0 = init_sig.connect(init_nas_context);
  sig_id1 = send_sig.connect(ready_to_send_nas_message);
  sig_id2 = handle_sig.connect(wait_for_nas_message);

  if (signal(SIGALRM, CompletionHandler) == SIG_ERR) {
    fprintf(stderr, "signal() error\n");
    exit(-1);
  }
  alarm(RESPONSE_WAIT_TIME);
  return;
}

contester_nas::~contester_nas()
{
  init_sig.disconnect(sig_id0);
  send_sig.disconnect(sig_id1);
  handle_sig.disconnect(sig_id2);
  return;
}

void contester_nas::init()
{
  read_test_scripts();

  msg_head = contest_msg_list.begin();
  msg_head++;

  cout << "::Contester Inited::" << endl;
}

contester_nas* contester_nas::get_instance(void)
{
  pthread_mutex_lock(&contester_nas_instance_locker);
  if (NULL == m_instance) {
    m_instance = new contester_nas();
  }
  pthread_mutex_unlock(&contester_nas_instance_locker);
  return (m_instance);
}

void contester_nas::cleanup(void)
{
  pthread_mutex_lock(&contester_nas_instance_locker);
  if (NULL != m_instance) {
    delete m_instance;
    m_instance = NULL;
  }
  pthread_mutex_unlock(&contester_nas_instance_locker);
}

void contester_nas::CompletionHandler(int signum)
{
  if (signum == SIGALRM){
    srsran::console("*** Contester ***  No response from UE\n");
  }

  if(m_instance->msg_head != m_instance->contest_msg_list.end() && m_instance->msg_head->sequence_id !=0){
    srsran::console("*** Contester *** ::Path::0x%x\n", m_instance->msg_head->contest_msg_type);
    m_instance->nas_ctx->m_contester_nas->send_sig.emit(m_instance->msg_head);
    m_instance->msg_head++;
  }
  else if (m_instance->msg_head->sequence_id == 0){
    srsran::console("*** Contester *** Test Case Finished\n");
    contester_nas::completion_signal_disable();
    contester_nas::completion_signal_init(m_instance->msg_head->parameter);
    if(LIBLTE_MME_MSG_TYPE_ATTACH_REQUEST == m_instance->msg_head->contest_msg_type || LIBLTE_MME_SECURITY_HDR_TYPE_SERVICE_REQUEST == m_instance->msg_head->contest_msg_type){
      srsran::console("*** Contester *** TEST RESULT: %s\n", m_instance->msg_head->is_success == 1 ? "Success":"Fail");
    }
    else{
      srsran::console("*** Contester *** TEST RESULT: Fail\n");
    }
    exit(2);
  }
}

void contester_nas::completion_signal_init(int wait_time)
{
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGALRM);
  sigprocmask(SIG_UNBLOCK, &set, NULL);
  if (signal(SIGALRM, CompletionHandler) == SIG_ERR) {
    fprintf(stderr, "signal() error\n");
    exit(-1);
  }
  alarm(wait_time);
}

void contester_nas::completion_signal_disable()
{
  alarm(0);
}

bool contester_nas::read_test_scripts()
{
	json jMessage;
	std::ifstream jfile("../test_scripts/S6_7.json");
	if (!jfile.is_open())
	{
		srsran::console("*** Contester *** :: Open test scripts failed.\n");
		return -1;
	}
	try
	{
    jMessage = json::parse(jfile);
		if (jMessage.is_discarded())
		{
			srsran::console("*** Contester *** :: Parse json data failed.\n");
			return -1;
		}
    
    uint8_t seq_id_int;
    std::string message_str;
    uint8_t is_replayed;
    uint8_t parameter;
    uint8_t is_reset_mac;
    uint8_t is_success = 0;

    int     temp_hex;
    uint8_t msg_type_hex;

    contest_msg_list.clear();
  
    for (size_t l = 0; l<jMessage.size(); l++) 
    {
      seq_id_int  = jMessage[l].at("Step");
      message_str = jMessage[l].at("Message");
      is_replayed = jMessage[l].at("Replayed");
      try{
        parameter = jMessage[l].at("Parameter")[0];
        is_reset_mac = (uint8_t)(jMessage[l].at("Parameter")[1]);
      }catch(nlohmann::detail::exception& e){
        is_reset_mac = 0;
      }
      if(seq_id_int==0){
        is_success = jMessage[l].at("Result");
      }
      
      // change str to int value
      std::stringstream convert;
      convert << get_msg_type_hex(message_str.c_str());
      convert >> std::hex >> temp_hex;
      cout <<"::Read_test_scripts:: "  << message_str<<" :: " << get_msg_type_hex(message_str.c_str())<<" :: "<<temp_hex << endl;
      msg_type_hex = (uint8_t)temp_hex;

      contest_msg_class contest_msg = contest_msg_class(seq_id_int, msg_type_hex, is_replayed, parameter, is_reset_mac, is_success);
      contest_msg_list.push_back(contest_msg);
    }
	}
	catch (std::exception& e)
	{
		srsran::console(e.what());

	}
	jMessage.clear();

  return 0;
}

void contester_nas::set_parameter_before_pack_message(uint8_t msg_type, ...)
{
  uint8 current_msg_type = msg_type;
  uint8_t* param;
  va_list p_arg;
  va_start(p_arg, msg_type);

  switch (current_msg_type) {
    case LIBLTE_MME_MSG_TYPE_IDENTITY_REQUEST:
      srsran::console("*** Contester *** :: Set Parameter: Identity Request\n");
      param = va_arg(p_arg, uint8_t*);
      *param = m_instance->msg_head->parameter;
      break;

    case LIBLTE_MME_MSG_TYPE_AUTHENTICATION_REQUEST:
      srsran::console("*** Contester *** :: Set Parameter: Authentication Request\n");
      param = va_arg(p_arg, uint8_t*);
      *param = m_instance->msg_head->parameter;
      break;

    case LIBLTE_MME_MSG_TYPE_SECURITY_MODE_COMMAND:
      srsran::console("*** Contester *** :: Set Parameter: Security Mode Command\n");
      param = va_arg(p_arg, uint8_t*);
      *param = m_instance->msg_head->parameter;
      break;

    case LIBLTE_MME_MSG_TYPE_ESM_INFORMATION_REQUEST:
      srsran::console("*** Contester *** :: Set Parameter: ESM Information Request\n");
      param = va_arg(p_arg, uint8_t*);
      *param = m_instance->msg_head->parameter;
      break;

    case LIBLTE_MME_MSG_TYPE_ATTACH_ACCEPT:
      srsran::console("*** Contester *** :: Set Parameter: Attach Accept\n");
      param = va_arg(p_arg, uint8_t*);
      *param = m_instance->msg_head->parameter;
      break;

    case LIBLTE_MME_MSG_TYPE_EMM_INFORMATION:
      srsran::console("*** Contester *** :: Set Parameter: EMM Information\n");
      param = va_arg(p_arg, uint8_t*);
      *param = m_instance->msg_head->parameter;
      break;

    case LIBLTE_MME_MSG_TYPE_GUTI_REALLOCATION_COMMAND:
      srsran::console("*** Contester *** :: Set Parameter: GUTI Reallocation Command\n");
      param = va_arg(p_arg, uint8_t*);
      *param = m_instance->msg_head->parameter;
      break;

    case LIBLTE_MME_MSG_TYPE_DOWNLINK_NAS_TRANSPORT:
      srsran::console("*** Contester *** :: Set Parameter: Downlink NAS Transport\n");
      param = va_arg(p_arg, uint8_t*);
      *param = m_instance->msg_head->parameter;
      break;

    case LIBLTE_MME_MSG_TYPE_ATTACH_REJECT:
      srsran::console("*** Contester *** :: Set Parameter: Attach Reject\n");
      param = va_arg(p_arg, uint8_t*);
      *param = m_instance->msg_head->parameter;
      break;

    case LIBLTE_MME_MSG_TYPE_TRACKING_AREA_UPDATE_REJECT:
      srsran::console("*** Contester *** :: Set Parameter: Tracking Area Update Reject\n");
      param = va_arg(p_arg, uint8_t*);
      *param = m_instance->msg_head->parameter;
      break;

    case LIBLTE_MME_MSG_TYPE_SERVICE_REJECT:
      srsran::console("*** Contester *** :: Set Parameter: Service Reject\n");
      param = va_arg(p_arg, uint8_t*);
      *param = m_instance->msg_head->parameter;
      break;

    default:
      break;
  }

  va_end(p_arg);
}

void contester_nas::set_mac_after_pack_message(uint8_t msg_type, ...)
{
  uint8 current_msg_type = msg_type;
  uint8_t* param;
  va_list p_arg;
  if(m_instance->msg_head->is_reset_mac!=0){
    va_start(p_arg, msg_type);
    param = va_arg(p_arg, uint8_t*);
    param[0] = 0xFF;
    param[1] = 0xFF;
    param[2] = 0xFF;
    param[3] = 0xFF;
  }
  va_end(p_arg);
}

void contester_nas::init_nas_context(nas* _nas_ctx, srsran::byte_buffer_t* _nas_buffer, s1ap_interface_nas* _s1ap)
{
  m_instance->nas_ctx = _nas_ctx;
  m_instance->nas_buffer = _nas_buffer;
  m_instance->s1ap = _s1ap;

  if(m_instance->msg_head != m_instance->contest_msg_list.end() && m_instance->msg_head->sequence_id !=0){
    srsran::console("*** Contester *** ::Path::0x%x\n", m_instance->msg_head->contest_msg_type);
    m_instance->nas_ctx->m_contester_nas->send_sig.emit(m_instance->msg_head);
    m_instance->msg_head++;
  }
  else if (m_instance->msg_head->sequence_id == 0){
    srsran::console("*** Contester *** Test Case Finished\n");
    contester_nas::completion_signal_disable();
    contester_nas::completion_signal_init(m_instance->msg_head->parameter);
    if(LIBLTE_MME_MSG_TYPE_ATTACH_REQUEST == m_instance->msg_head->contest_msg_type || LIBLTE_MME_SECURITY_HDR_TYPE_SERVICE_REQUEST == m_instance->msg_head->contest_msg_type){
      srsran::console("*** Contester *** TEST RESULT: %s\n", m_instance->msg_head->is_success==1?"Success":"Fail");
    }
    else{
      srsran::console("*** Contester *** TEST RESULT: Fail\n");
    }
    exit(2);
  }
}

void contester_nas::ready_to_send_nas_message(list<contest_msg_class>::iterator msg_class)
{
  srsran::unique_byte_buffer_t      nas_tx;
  nas_tx = srsran::make_byte_buffer();
  if (nas_tx == nullptr) {
    srsran::console("*** Contester *** Couldn't allocate PDU in %s().\n", __FUNCTION__);
    return;
  }
  m_instance->nas_buffer = nas_tx.get();

  switch (msg_class->contest_msg_type) {
    case LIBLTE_MME_MSG_TYPE_IDENTITY_REQUEST:
      // Send a Identity Request Message
      m_instance->nas_ctx->pack_identity_request(m_instance->nas_buffer);
      m_instance->s1ap->send_downlink_nas_transport(
        m_instance->nas_ctx->m_ecm_ctx.enb_ue_s1ap_id, m_instance->nas_ctx->m_ecm_ctx.mme_ue_s1ap_id, m_instance->nas_buffer, m_instance->nas_ctx->m_ecm_ctx.enb_sri);
      srsran::console("*** Contester *** MME>>>UE:: Identity Request\n");
      break;

    case LIBLTE_MME_MSG_TYPE_AUTHENTICATION_REQUEST:
      // Send a Authentication Request Message
      m_instance->nas_ctx->pack_authentication_request(m_instance->nas_buffer);
      m_instance->s1ap->send_downlink_nas_transport(
        m_instance->nas_ctx->m_ecm_ctx.enb_ue_s1ap_id, m_instance->nas_ctx->m_ecm_ctx.mme_ue_s1ap_id, m_instance->nas_buffer, m_instance->nas_ctx->m_ecm_ctx.enb_sri);
      srsran::console("*** Contester *** MME>>>UE:: Authentication Request\n");
      break;

    case LIBLTE_MME_MSG_TYPE_SECURITY_MODE_COMMAND:
      // Send a Security Mode Command Message
      m_instance->nas_ctx->pack_security_mode_command(m_instance->nas_buffer);
      m_instance->s1ap->send_downlink_nas_transport(
        m_instance->nas_ctx->m_ecm_ctx.enb_ue_s1ap_id, m_instance->nas_ctx->m_ecm_ctx.mme_ue_s1ap_id, m_instance->nas_buffer, m_instance->nas_ctx->m_ecm_ctx.enb_sri);
      srsran::console("*** Contester *** MME>>>UE:: Security Mode Command\n");
      break;

    case LIBLTE_MME_MSG_TYPE_ESM_INFORMATION_REQUEST:
      // Send a ESM Information Request Message
      m_instance->nas_ctx->pack_esm_information_request(m_instance->nas_buffer);
      m_instance->s1ap->send_downlink_nas_transport(
        m_instance->nas_ctx->m_ecm_ctx.enb_ue_s1ap_id, m_instance->nas_ctx->m_ecm_ctx.mme_ue_s1ap_id, m_instance->nas_buffer, m_instance->nas_ctx->m_ecm_ctx.enb_sri);
      srsran::console("*** Contester *** MME>>>UE:: ESM Information Request\n");
      break;

    case LIBLTE_MME_MSG_TYPE_ATTACH_ACCEPT:
      // Send a Attach Accept Message
      // nas_ctx->pack_attach_accept(nas_buffer);
      srsran::console("*** Contester *** MME>>>UE:: Attach Accept\n");
      break;

    case LIBLTE_MME_MSG_TYPE_EMM_INFORMATION:
      // Send a EMM Information Message
      m_instance->nas_ctx->pack_emm_information(m_instance->nas_buffer);
      m_instance->s1ap->send_downlink_nas_transport(
        m_instance->nas_ctx->m_ecm_ctx.enb_ue_s1ap_id, m_instance->nas_ctx->m_ecm_ctx.mme_ue_s1ap_id, m_instance->nas_buffer, m_instance->nas_ctx->m_ecm_ctx.enb_sri);
      srsran::console("*** Contester *** MME>>>UE:: EMM Information\n");
      break;

    case LIBLTE_MME_MSG_TYPE_GUTI_REALLOCATION_COMMAND:
      // Send a GUTI Reallocation Command Message
      m_instance->nas_ctx->pack_guti_reallocation_command(m_instance->nas_buffer);
      m_instance->s1ap->send_downlink_nas_transport(
        m_instance->nas_ctx->m_ecm_ctx.enb_ue_s1ap_id, m_instance->nas_ctx->m_ecm_ctx.mme_ue_s1ap_id, m_instance->nas_buffer, m_instance->nas_ctx->m_ecm_ctx.enb_sri);
      srsran::console("*** Contester *** MME>>>UE:: GUTI Reallocation Command\n");
      break;

    case LIBLTE_MME_MSG_TYPE_DOWNLINK_NAS_TRANSPORT:
      // Send a Downlink NAS Transport Message
      m_instance->nas_ctx->pack_downlink_nas_transport_msg(m_instance->nas_buffer);
      m_instance->s1ap->send_downlink_nas_transport(
        m_instance->nas_ctx->m_ecm_ctx.enb_ue_s1ap_id, m_instance->nas_ctx->m_ecm_ctx.mme_ue_s1ap_id, m_instance->nas_buffer, m_instance->nas_ctx->m_ecm_ctx.enb_sri);
      srsran::console("*** Contester *** MME>>>UE:: Downlink NAS Transport\n");
      break;

    case LIBLTE_MME_MSG_TYPE_AUTHENTICATION_REJECT:
      // Send an Authentication Reject Message
      m_instance->nas_ctx->pack_authentication_reject(m_instance->nas_buffer);
      m_instance->s1ap->send_downlink_nas_transport(
        m_instance->nas_ctx->m_ecm_ctx.enb_ue_s1ap_id, m_instance->nas_ctx->m_ecm_ctx.mme_ue_s1ap_id,m_instance->nas_buffer, m_instance->nas_ctx->m_ecm_ctx.enb_sri);
      srsran::console("*** Contester *** MME>>>UE:: Authentication Reject\n");
      break;

    case  LIBLTE_MME_MSG_TYPE_ATTACH_REJECT:
      // Send an Attach Reject Message
      m_instance->nas_ctx->pack_attach_reject(m_instance->nas_buffer, 0);
      m_instance->s1ap->send_downlink_nas_transport(
        m_instance->nas_ctx->m_ecm_ctx.enb_ue_s1ap_id, m_instance->nas_ctx->m_ecm_ctx.mme_ue_s1ap_id,m_instance->nas_buffer, m_instance->nas_ctx->m_ecm_ctx.enb_sri);
      srsran::console("*** Contester *** MME>>>UE:: Attach Reject\n");
      break;

    case  LIBLTE_MME_MSG_TYPE_TRACKING_AREA_UPDATE_REJECT:
      // Send a Tracking Area Update Reject Message
      m_instance->nas_ctx->pack_tracking_area_update_reject(m_instance->nas_buffer, 0);
      m_instance->s1ap->send_downlink_nas_transport(
        m_instance->nas_ctx->m_ecm_ctx.enb_ue_s1ap_id, m_instance->nas_ctx->m_ecm_ctx.mme_ue_s1ap_id,m_instance->nas_buffer, m_instance->nas_ctx->m_ecm_ctx.enb_sri);
      srsran::console("*** Contester *** MME>>>UE:: Tracking Area Update Reject\n");
      break;

    case  LIBLTE_MME_MSG_TYPE_SERVICE_REJECT:
      // Send a Service Reject Message
      m_instance->nas_ctx->pack_service_reject(m_instance->nas_buffer, 0);
      m_instance->s1ap->send_downlink_nas_transport(
        m_instance->nas_ctx->m_ecm_ctx.enb_ue_s1ap_id, m_instance->nas_ctx->m_ecm_ctx.mme_ue_s1ap_id,m_instance->nas_buffer, m_instance->nas_ctx->m_ecm_ctx.enb_sri);
      srsran::console("*** Contester *** MME>>>UE:: Service Reject\n");
      break;

    case 0x00:
      if(m_instance->nas_buffer_replay != NULL){
        m_instance->s1ap->send_downlink_nas_transport(
          m_instance->nas_ctx->m_ecm_ctx.enb_ue_s1ap_id, m_instance->nas_ctx->m_ecm_ctx.mme_ue_s1ap_id, m_instance->nas_buffer_replay.get(), m_instance->nas_ctx->m_ecm_ctx.enb_sri);
        srsran::console("*** Contester *** MME>>>UE:: REPLAYED MESSAGE\n");
      }
      break;

    default:
      srsran::console("*** Contester *** MME>>>UE:: UNKNOWN MESSAGE %x\n", msg_class->contest_msg_type);
      break;
  }

  if (msg_class->is_replayed == 1){
    m_instance->nas_buffer_replay = srsran::make_byte_buffer();
    m_instance->nas_buffer_replay->N_bytes = m_instance->nas_buffer->size();
    memcpy(m_instance->nas_buffer_replay->msg, m_instance->nas_buffer->msg, m_instance->nas_buffer->size());
  }

  contester_nas::completion_signal_disable();
  contester_nas::completion_signal_init(RESPONSE_WAIT_TIME);
}

void contester_nas::wait_for_nas_message(uint8_t msg_type, nas* nas_ctx, srsran::byte_buffer_t* nas_buffer)
{
    switch (msg_type) {
    case LIBLTE_MME_MSG_TYPE_ATTACH_REQUEST:
      srsran::console("*** Contester *** MME<<<UE:: Attach Resquest\n");
      break;
    case LIBLTE_MME_MSG_TYPE_IDENTITY_RESPONSE:
      nas_ctx->handle_identity_response(nas_buffer);
      srsran::console("*** Contester *** MME<<<UE:: Identity Response\n");
      break;
    case LIBLTE_MME_MSG_TYPE_AUTHENTICATION_RESPONSE:
      srsran::console("*** Contester *** MME<<<UE:: Authentication Response\n");
      nas_ctx->handle_authentication_response(nas_buffer);
      break;
    // Authentication failure with the option sync failure can be sent not integrity protected
    case LIBLTE_MME_MSG_TYPE_AUTHENTICATION_FAILURE:
      srsran::console("*** Contester *** MME<<<UE:: Authentication Failure\n");
      nas_ctx->handle_authentication_failure(nas_buffer);
      break;
    // Detach request can be sent not integrity protected when "power off" option is used
    case LIBLTE_MME_MSG_TYPE_DETACH_REQUEST:
      srsran::console("*** Contester *** MME<<<UE:: Detach Request\n");
      nas_ctx->handle_detach_request(nas_buffer);
      break;
    case LIBLTE_MME_MSG_TYPE_SECURITY_MODE_COMPLETE:
      srsran::console("*** Contester *** MME<<<UE:: Security Mode Complete\n");
      nas_ctx->handle_security_mode_complete(nas_buffer);
      break;
    case LIBLTE_MME_MSG_TYPE_ATTACH_COMPLETE:
      srsran::console("*** Contester *** MME<<<UE:: Attach Complete\n");
      nas_ctx->handle_attach_complete(nas_buffer);
      break;
    case LIBLTE_MME_MSG_TYPE_ESM_INFORMATION_RESPONSE:
      srsran::console("*** Contester *** MME<<<UE:: ESM Information Response\n");
      nas_ctx->handle_esm_information_response(nas_buffer);
      break;
    case LIBLTE_MME_MSG_TYPE_TRACKING_AREA_UPDATE_REQUEST:
      srsran::console("*** Contester *** MME<<<UE:: Tracking Area Update Request\n");
      nas_ctx->handle_tracking_area_update_request(nas_buffer);
      break;
    case LIBLTE_MME_MSG_TYPE_PDN_CONNECTIVITY_REQUEST:
      srsran::console("*** Contester *** MME<<<UE:: PDN Connectivity Request\n");
      nas_ctx->handle_pdn_connectivity_request(nas_buffer);
      break;
    case LIBLTE_MME_SECURITY_HDR_TYPE_SERVICE_REQUEST:
      srsran::console("*** Contester *** MME<<<UE:: Service Request\n");
      break;

    default:
      srsran::console("*** Contester *** Unhandled NAS integrity protected message %s\n", liblte_nas_msg_type_to_string(msg_type));
  }

  if(m_instance->msg_head != m_instance->contest_msg_list.end() && m_instance->msg_head->sequence_id !=0){
    srsran::console("*** Contester *** ::Path::0x%x\n", m_instance->msg_head->contest_msg_type);
    m_instance->nas_ctx->m_contester_nas->send_sig.emit(m_instance->msg_head);
    m_instance->msg_head++;
  }
  else if (m_instance->msg_head->sequence_id == 0){
    srsran::console("*** Contester *** Test Case Finished\n");
    contester_nas::completion_signal_disable();
    contester_nas::completion_signal_init(m_instance->msg_head->parameter);
    if(msg_type == m_instance->msg_head->contest_msg_type){
      srsran::console("*** Contester *** TEST RESULT: %s\n", m_instance->msg_head->is_success==1?"Success":"Fail");
    }
    else if(LIBLTE_MME_MSG_TYPE_ATTACH_REQUEST == m_instance->msg_head->contest_msg_type || LIBLTE_MME_SECURITY_HDR_TYPE_SERVICE_REQUEST == m_instance->msg_head->contest_msg_type){
      srsran::console("*** Contester *** TEST RESULT: %s\n", m_instance->msg_head->is_success==1?"Success":"Fail");
    }
    else{
      srsran::console("*** Contester *** TEST RESULT: Fail\n");
    }
    exit(2);
  }
}

char* contester_nas::get_msg_type_name(const char* hex_str)
{
  for (int i = 0; i < MSG_TYPE_N; i++) {
    if (strncmp(hex_str, msg_type_name_hex_map[i][1], strlen(hex_str)) == 0) {
      return msg_type_name_hex_map[i][0];
    }
  }
  return NULL;
}

char* contester_nas::get_msg_type_hex(const char* name)
{
  for (int i = 0; i < MSG_TYPE_N; i++) {
    if (strncmp(name, msg_type_name_hex_map[i][0], strlen(name)) == 0) {
      return msg_type_name_hex_map[i][1];
    }
  }
  return NULL;
}

char* contester_nas::get_msg_type_str_from_uint(const uint8_t cur_msg_type)
{
  std::stringstream sstream;
  sstream << "0x" << std::setfill('0') << std::setw(2) << std::hex << (int)cur_msg_type;
  return get_msg_type_name(sstream.str().c_str());
}

} // namespace srsepc