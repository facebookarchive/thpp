#include <cstdlib>
#include <iostream>

int g_you_cannot_have_one_binary_with_two_versions_of_th_or_thpp_linked = 0;

namespace {
struct Trigger {
  Trigger() {
    if (g_you_cannot_have_one_binary_with_two_versions_of_th_or_thpp_linked) {
      std::cerr
          << "you can't have one binary with two versions of th or thpp linked"
          << std::endl;
      exit(1);
    }
    ++g_you_cannot_have_one_binary_with_two_versions_of_th_or_thpp_linked;
  }
};

Trigger g_trigger;
} // namespace
