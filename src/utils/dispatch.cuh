#ifndef UTILS_DISPATCH_CUH_
#define UTILS_DISPATCH_CUH_

/// @brief Generic functor for choosing an unsigned template parameter at
/// runtime. Selects the smallest integer greather than or equal to the
/// `desired` number and launches a provided templated functor with it (calls
/// the `operator()` of the provided `Launch` Functor, which must have a trivial
/// constructor).
/// @tparam Elem
/// @tparam ...Cons
template <template <unsigned> class Launch, unsigned Elem, unsigned... Cons>
struct ListDispatcher {
    template <typename... Args>
    static void dispatch(unsigned desired, Args&&... args)
    {
        if (desired <= Elem) {
            Launch<Elem> {}(std::forward<Args>(args)...);
        } else {
            ListDispatcher<Launch, Cons...>::dispatch(
                desired, std::forward<Args>(args)...);
        }
    }
};
/// @brief Base case of the recursively defined `ListDispatcher` template, where
/// the list contains only a single entry, making it equivalent to simply
/// calling `operator()` on the provided `Launch` functor
/// @tparam Last
template <template <unsigned> class Launch, unsigned Last>
struct ListDispatcher<Launch, Last> {
    template <typename... Args>
    static void dispatch(unsigned runtime_depth, Args&&... args)
    {
        Launch<Last> {}(std::forward<Args>(args)...);
    }
};

#endif // UTILS_DISPATCH_CUH_