#include <memory>
#include <iostream>
#include <cstdlib>

class Foo
{
public:
    Foo()
    {
        std::cout << "Constructed Foo object: " << this << std::endl;
    }

    ~Foo()
    {
        std::cout << "Destroying Foo object: " << this << std::endl;
    }
};


int main()
{
    // use malloc to allocate, and then use free() to free it
    // what do you notice?
    std::cout << "===== malloc ====="  << std::endl;
    Foo *f1 = (Foo *) malloc(sizeof(Foo));
    free(f1);
    std::cout << "==================" << std::endl;

    // use new to allocate, then use delete to free it
    // what do you notice?
    std::cout << "===== new ====="  << std::endl;
    Foo *f2 = new Foo();
    delete f2;
    std::cout << "=================" << std::endl;

    // create a new std::shared_ptr<Foo> f3, and use std::make_shared<Foo> to make a new one.
    // what do you notice?
    std::cout << "===== shared ====="  << std::endl;
    std::shared_ptr<Foo> f3 = std::make_shared<Foo>();
    std::cout << "=================" << std::endl;
    return 0;
}

