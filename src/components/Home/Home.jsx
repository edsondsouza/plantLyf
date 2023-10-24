import React from "react";
import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="mx-auto w-full max-w-7xl">
      <aside className="relative overflow-hidden text-black rounded-lg sm:mx-16 mx-2 sm:py-16">
        <div className="relative z-10 max-w-screen-xl px-4  pb-20 pt-10 sm:py-24 mx-auto sm:px-6 lg:px-8">
          <div className="max-w-xl sm:mt-1 mt-80 space-y-8 text-center sm:text-right sm:ml-auto">
            <h2 className="text-4xl font-bold sm:text-5xl">
              <span className="text-green-600">Explore Now</span>
              <span className="hidden sm:block text-4xl">Lorem Ipsum</span>
            </h2>

            <Link
              className="inline-flex text-white items-center px-6 py-3 font-medium bg-green-700 rounded-lg hover:opacity-75"
              to="/"
            >
              &nbsp; Read Documentation
            </Link>
          </div>
        </div>

        <div className="absolute inset-0 w-full sm:my-20 sm:pt-1 pt-12 h-full ">
          <img
            className="w-120"
            src="https://images.pexels.com/photos/18620459/pexels-photo-18620459/free-photo-of-photo-of-a-senior-farmer-sowing-in-a-rice-field.jpeg?auto=compress&cs=tinysrgb&w=600"
            alt="image1"
          />
        </div>
      </aside>

      <div className="grid  place-items-center sm:mt-20">
        <img
          className="sm:w-96 w-120 "
          src="https://images.pexels.com/photos/11070640/pexels-photo-11070640.jpeg?auto=compress&cs=tinysrgb&w=600"
          alt="image2"
        />
      </div>

      <h1 className="text-center text-2xl sm:text-5xl py-10 font-medium">
        Lorem Ipsum Yojo
      </h1>
    </div>
  );
}
