
import { motion } from "framer-motion";
import Navbar from "../components/Navbar";


const DashboardPage = () => {
	
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="text-center"
    >
      <Navbar />


      <h2 className="text-3xl text-white font-bold mb-6">
        Recommendations are on the way!
      </h2>



    </motion.div>
  );
};

export default DashboardPage;
