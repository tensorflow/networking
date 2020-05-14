#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_TAG_FACTORY_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_TAG_FACTORY_H_

#include "seastar/core/channel.hh"
#include "tensorflow_networking/seastar/seastar_client_tag.h"
#include "tensorflow_networking/seastar/seastar_server_tag.h"
#include "tensorflow_networking/seastar/seastar_worker_service.h"

#include "seastar/core/temporary_buffer.hh"

namespace tensorflow {

class SeastarTagFactory {
 public:
  explicit SeastarTagFactory(SeastarWorkerService* worker_service);
  virtual ~SeastarTagFactory() {}

  SeastarClientTag* CreateSeastarClientTag(
      seastar::temporary_buffer<char>& header);

  SeastarServerTag* CreateSeastarServerTag(
      seastar::temporary_buffer<char>& header,
      seastar::channel* seastar_channel);

 private:
  SeastarWorkerService* worker_service_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_TAG_FACTORY_H_
