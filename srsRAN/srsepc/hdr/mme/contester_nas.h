#ifndef CONTESTER_NAS_H
#define CONTESTER_NAS_H

#include <stdbool.h>
#include <unistd.h>

#include "s1ap_common.h"

#include <fstream>
#include <list>
#include <time.h>

#include "json.hpp"
#include "SimpleSignal.h"

using namespace std;

namespace srsepc {

#define TEST_SUCCESS_CODE 1
#define TEST_FAIL_CODE 0

#define MSG_TYPE_N 33

using json = nlohmann::json;

class contest_msg_class
{

public:
  contest_msg_class() {}
  contest_msg_class(uint8_t _seq_id, uint8_t _msg_type, uint8_t _is_replayed, uint8_t _parameter, uint8_t _is_reset_mac, uint8_t _is_success) : sequence_id(_seq_id), contest_msg_type(_msg_type), is_replayed(_is_replayed), parameter(_parameter), is_reset_mac(_is_reset_mac), is_success(_is_success) {}

  uint8_t sequence_id;
  uint8_t contest_msg_type;
  uint8_t is_replayed;
  uint8_t parameter;
  uint8_t is_reset_mac;
  uint8_t is_success;
  int     id_type_idx;
  int     sec_hdr_type_idx;

};

class contester_nas
{

public:
  srsran::byte_buffer_t* replayed_nas_msg;
  Simple::Signal<void (nas* , srsran::byte_buffer_t*, s1ap_interface_nas*)> init_sig;
  Simple::Signal<void (list<contest_msg_class>::iterator)> send_sig;
  Simple::Signal<void (uint8_t, nas*, srsran::byte_buffer_t*)> handle_sig;


  uint8_t           contest_msg;
  list<contest_msg_class> contest_msg_list;
  list<contest_msg_class>::iterator msg_head;


  static contester_nas* m_instance;
  static contester_nas* get_instance(void);
  static void           cleanup(void);

  nas* nas_ctx;
  srsran::byte_buffer_t* nas_buffer;
  srsran::unique_byte_buffer_t nas_buffer_replay = NULL;
  s1ap_interface_nas* s1ap;

  static void           init_nas_context(nas* nas_ctx, srsran::byte_buffer_t* nas_buffer, s1ap_interface_nas* s1ap);
  static void           wait_for_nas_message(uint8_t msg_type, nas* nas_ctx, srsran::byte_buffer_t* nas_buffer);
  static void           ready_to_send_nas_message(list<contest_msg_class>::iterator msg_class);
  static void           CompletionHandler(int signum);
  static void           completion_signal_init(int wait_time);
  static void           completion_signal_disable();


  void                  init();
  bool                  read_test_scripts();

  void                  set_parameter_before_pack_message(uint8_t msg_type, ...);
  void                  set_mac_after_pack_message(uint8_t msg_type, ...);



private:
  contester_nas();
  virtual ~contester_nas();


  char* get_msg_type_name(const char* hex_str);
  char* get_msg_type_hex(const char* name);
  char* get_msg_type_str_from_uint(const uint8_t cur_msg_type);

  #define RESPONSE_WAIT_TIME 3   // second

  char  msg_type_name_hex_map[MSG_TYPE_N][2][100] = {
      {"REPLAYED_MESSAGE", "0x00"},
      {"SERVICE_REQUEST","0x0C"},
      {"ATTACH_REQUEST", "0x41"},
      {"ATTACH_ACCEPT", "0x42"},
      {"ATTACH_COMPLETE", "0x43"},
      {"ATTACH_REJECT", "0x44"},
      {"DETACH_REQUEST", "0x45"},
      {"DETACH_ACCEPT", "0x46"},
      {"TRACKING_AREA_UPDATE_REQUEST", "0x48"},
      {"TRACKING_AREA_UPDATE_ACCEPT", "0x49"},
      {"TRACKING_AREA_UPDATE_COMPLETE", "0x4A"},
      {"TRACKING_AREA_UPDATE_REJECT", "0x4B"},
      {"EXTENDED_SERVICE_REQUEST", "0x4C"},
      {"SERVICE_REJECT", "0x4E"},
      {"GUTI_REALLOCATION_COMMAND", "0x50"},
      {"GUTI_REALLOCATION_COMPLETE", "0x51"},
      {"AUTHENTICATION_REQUEST", "0x52"},
      {"AUTHENTICATION_RESPONSE", "0x53"},
      {"AUTHENTICATION_REJECT", "0x54"},
      {"AUTHENTICATION_FAILURE", "0x5C"},
      {"IDENTITY_REQUEST", "0x55"},
      {"IDENTITY_RESPONSE", "0x56"},
      {"SECURITY_MODE_COMMAND", "0x5D"},
      {"SECURITY_MODE_COMPLETE", "0x5E"},
      {"SECURITY_MODE_REJECT", "0x5F"},
      {"EMM_STATUS", "0x60"},
      {"EMM_INFORMATION", "0x61"},
      {"DOWNLINK_NAS_TRANSPORT", "0x62"},
      {"UPLINK_NAS_TRANSPORT", "0x63"},
      {"ESM_INFORMATION_REQUEST", "0xD9"},
      {"ESM_INFORMATION_RESPONSE", "0xDA"},
      {"NOTIFICATION", "0xDB"},
      {"ESM_STATUS", "0xE8"}
      };
};


} // namespace srsepc

#endif